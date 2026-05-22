"""Pydantic v2 request / response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Request
# ------------------------------------------------------------------

class DetectionRequest(BaseModel):
    """Metadata accompanying uploaded raster files."""
    scene_id: Optional[str] = None
    before_date: Optional[datetime] = None
    after_date: Optional[datetime] = None
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_area_ha: float = Field(default=1.0, gt=0.0)


# ------------------------------------------------------------------
# Response
# ------------------------------------------------------------------

class Centroid(BaseModel):
    lon: float
    lat: float


class AlertResponse(BaseModel):
    alert_id: str
    scene_id: str
    detected_at: datetime
    severity: Literal["low", "medium", "high"]
    area_ha: float
    centroid: Centroid
    geometry: dict[str, Any]   # GeoJSON geometry


class DetectionResponse(BaseModel):
    job_id: str
    scene_id: str
    status: Literal["completed", "failed"]
    alert_count: int
    alerts: list[AlertResponse]
    processing_time_s: float


class AlertsQueryResponse(BaseModel):
    total: int
    alerts: list[AlertResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
