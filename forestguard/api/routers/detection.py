"""
Detection router.

POST /detect — upload before/after raster pair → run inference → return alerts.
"""

from __future__ import annotations

import io
import logging
import time
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from ..dependencies import AlertGenDep, AlertStoreDep, ModelDep
from ..schemas import DetectionResponse, AlertResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/detect", tags=["Detection"])

_MAX_BYTES = 512 * 1024 * 1024   # 512 MB


@router.post("", response_model=DetectionResponse, status_code=status.HTTP_200_OK)
async def run_detection(
    before_file: UploadFile = File(..., description="Before scene: fused SAR+optical GeoTIFF (C,H,W)"),
    after_file: UploadFile = File(..., description="After scene: fused SAR+optical GeoTIFF (C,H,W)"),
    scene_id: str = Form(default=""),
    confidence_threshold: float = Form(default=0.5, ge=0.0, le=1.0),
    min_area_ha: float = Form(default=1.0, gt=0.0),
    before_date: str = Form(default=""),
    after_date: str = Form(default=""),
    model: ModelDep = None,
    alert_gen: AlertGenDep = None,
    alert_store: AlertStoreDep = None,
) -> DetectionResponse:
    job_id = str(uuid.uuid4())
    scene_id = scene_id or job_id
    t0 = time.perf_counter()

    try:
        before_arr, after_arr, transform, crs = await _load_rasters(
            before_file, after_file
        )
    except Exception as exc:
        logger.exception("Failed to load raster files")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not read raster files: {exc}",
        ) from exc

    device = next(model.parameters()).device
    before_t = torch.from_numpy(before_arr).unsqueeze(0).to(device)
    after_t = torch.from_numpy(after_arr).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(before_t, after_t).squeeze(0)   # (1, H, W)

    alert_gen.confidence_threshold = confidence_threshold
    alert_gen.min_area_ha = min_area_ha

    scene_dt = _parse_date(after_date) or datetime.now(timezone.utc)
    raw_alerts = alert_gen.generate(logits, transform, crs, scene_dt, scene_id)

    # Persist to store
    alert_store.extend(raw_alerts)

    elapsed = time.perf_counter() - t0
    alert_responses = [AlertResponse(**_coerce(a)) for a in raw_alerts]

    return DetectionResponse(
        job_id=job_id,
        scene_id=scene_id,
        status="completed",
        alert_count=len(alert_responses),
        alerts=alert_responses,
        processing_time_s=round(elapsed, 3),
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

async def _load_rasters(before_file: UploadFile, after_file: UploadFile):
    before_bytes = await before_file.read(_MAX_BYTES)
    after_bytes = await after_file.read(_MAX_BYTES)

    with tempfile.TemporaryDirectory() as tmp:
        before_path = Path(tmp) / "before.tif"
        after_path = Path(tmp) / "after.tif"
        before_path.write_bytes(before_bytes)
        after_path.write_bytes(after_bytes)

        with rasterio.open(before_path) as src:
            before_arr = src.read().astype(np.float32)
            transform = src.transform
            crs = src.crs

        with rasterio.open(after_path) as src:
            after_arr = src.read().astype(np.float32)

    # Align spatial extents
    h = min(before_arr.shape[1], after_arr.shape[1])
    w = min(before_arr.shape[2], after_arr.shape[2])
    return before_arr[:, :h, :w], after_arr[:, :h, :w], transform, crs


def _parse_date(date_str: str) -> datetime | None:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return datetime.strptime(date_str, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _coerce(alert: dict) -> dict:
    """Ensure detected_at is a datetime object for Pydantic."""
    a = dict(alert)
    if isinstance(a.get("detected_at"), str):
        a["detected_at"] = datetime.fromisoformat(a["detected_at"])
    return a
