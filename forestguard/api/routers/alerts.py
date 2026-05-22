"""
Alerts router.

GET /alerts           — query all stored alerts (filterable)
GET /alerts/{id}      — fetch single alert
GET /alerts/export    — GeoJSON FeatureCollection export
"""

from __future__ import annotations

from typing import Literal, Optional
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import JSONResponse

from ..dependencies import AlertStoreDep
from ..schemas import AlertResponse, AlertsQueryResponse

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("", response_model=AlertsQueryResponse)
def list_alerts(
    severity: Optional[Literal["low", "medium", "high"]] = Query(default=None),
    min_area_ha: Optional[float] = Query(default=None, gt=0),
    scene_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    alert_store: AlertStoreDep = None,
) -> AlertsQueryResponse:
    filtered = alert_store

    if severity:
        filtered = [a for a in filtered if a["severity"] == severity]
    if min_area_ha is not None:
        filtered = [a for a in filtered if a["area_ha"] >= min_area_ha]
    if scene_id:
        filtered = [a for a in filtered if a["scene_id"] == scene_id]

    page = filtered[offset : offset + limit]
    return AlertsQueryResponse(
        total=len(filtered),
        alerts=[AlertResponse(**_coerce(a)) for a in page],
    )


@router.get("/export", response_class=JSONResponse)
def export_geojson(
    severity: Optional[Literal["low", "medium", "high"]] = Query(default=None),
    scene_id: Optional[str] = Query(default=None),
    alert_store: AlertStoreDep = None,
) -> dict:
    """Return all matching alerts as a GeoJSON FeatureCollection."""
    filtered = alert_store
    if severity:
        filtered = [a for a in filtered if a["severity"] == severity]
    if scene_id:
        filtered = [a for a in filtered if a["scene_id"] == scene_id]

    features = []
    for alert in filtered:
        features.append(
            {
                "type": "Feature",
                "geometry": alert["geometry"],
                "properties": {
                    k: v for k, v in alert.items() if k != "geometry"
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


@router.get("/{alert_id}", response_model=AlertResponse)
def get_alert(alert_id: str, alert_store: AlertStoreDep = None) -> AlertResponse:
    for alert in alert_store:
        if alert["alert_id"] == alert_id:
            return AlertResponse(**_coerce(alert))
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Alert {alert_id!r} not found.",
    )


def _coerce(alert: dict) -> dict:
    from datetime import datetime
    a = dict(alert)
    if isinstance(a.get("detected_at"), str):
        a["detected_at"] = datetime.fromisoformat(a["detected_at"])
    return a
