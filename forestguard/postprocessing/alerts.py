"""End-to-end alert generation: raw logits → filtered mask → GeoJSON alerts."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import torch
import geopandas as gpd
from rasterio.transform import Affine

from .filtering import MorphologicalFilter
from .vectorization import MaskVectorizer


_SEVERITY = {"low": 1.0, "medium": 10.0, "high": 100.0}


class AlertGenerator:
    """
    Orchestrates the full postprocessing chain:
      1. Sigmoid thresholding
      2. Morphological noise filtering
      3. Polygon vectorization
      4. Severity classification by area
      5. GeoJSON output
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        min_area_ha: float = 1.0,
        min_area_pixels: int = 100,
        morphology_kernel: int = 5,
        simplify_tolerance: float = 10.0,
        severity_thresholds: dict | None = None,
        output_crs: str = "EPSG:4326",
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.min_area_ha = min_area_ha
        self.severity_thresholds = severity_thresholds or _SEVERITY

        self.filter = MorphologicalFilter(
            kernel_size=morphology_kernel,
            min_area_pixels=min_area_pixels,
        )
        self.vectorizer = MaskVectorizer(
            simplify_tolerance=simplify_tolerance,
            output_crs=output_crs,
        )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(
        self,
        logits: torch.Tensor | np.ndarray,
        transform: Affine,
        src_crs: Any,
        scene_date: datetime | None = None,
        scene_id: str | None = None,
    ) -> list[dict]:
        """
        Parameters
        ----------
        logits     : (1, H, W) or (H, W) raw model output (pre-sigmoid)
        transform  : rasterio Affine of the prediction raster
        src_crs    : CRS of the prediction raster
        scene_date : acquisition date for alert metadata
        scene_id   : optional identifier for the input scene pair

        Returns
        -------
        List of alert dicts, each serialisable to JSON.
        """
        mask = self._threshold(logits)
        mask = self.filter(mask)

        gdf = self.vectorizer.vectorize(mask, transform, src_crs)

        # Area filter
        gdf = gdf[gdf["area_ha"] >= self.min_area_ha].reset_index(drop=True)

        if gdf.empty:
            return []

        gdf["severity"] = gdf["area_ha"].apply(self._classify_severity)
        gdf["alert_id"] = [str(uuid.uuid4()) for _ in range(len(gdf))]
        gdf["detected_at"] = (
            scene_date or datetime.now(timezone.utc)
        ).isoformat()
        gdf["scene_id"] = scene_id or "unknown"

        return self._to_records(gdf)

    def generate_from_file(
        self,
        prediction_path: str,
        scene_date: datetime | None = None,
        scene_id: str | None = None,
    ) -> list[dict]:
        """Load a GeoTIFF prediction raster and generate alerts."""
        import rasterio

        with rasterio.open(prediction_path) as src:
            logits = src.read(1)
            transform = src.transform
            crs = src.crs
        return self.generate(logits, transform, crs, scene_date, scene_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _threshold(self, logits: torch.Tensor | np.ndarray) -> np.ndarray:
        if isinstance(logits, torch.Tensor):
            logits = torch.sigmoid(logits).cpu().numpy()
        else:
            logits = 1.0 / (1.0 + np.exp(-logits))
        mask = (logits > self.confidence_threshold).astype(np.uint8)
        return mask.squeeze()

    def _classify_severity(self, area_ha: float) -> str:
        thresholds = sorted(self.severity_thresholds.items(), key=lambda x: x[1])
        severity = "low"
        for label, min_ha in thresholds:
            if area_ha >= min_ha:
                severity = label
        return severity

    @staticmethod
    def _to_records(gdf: gpd.GeoDataFrame) -> list[dict]:
        records = []
        for _, row in gdf.iterrows():
            records.append(
                {
                    "alert_id": row["alert_id"],
                    "scene_id": row["scene_id"],
                    "detected_at": row["detected_at"],
                    "severity": row["severity"],
                    "area_ha": row["area_ha"],
                    "centroid": {
                        "lon": row["centroid_lon"],
                        "lat": row["centroid_lat"],
                    },
                    "geometry": row["geometry"].__geo_interface__,
                }
            )
        return records
