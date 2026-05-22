"""Integration tests for the FastAPI endpoints (no real rasters — mock model)."""

from __future__ import annotations

import io
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from forestguard.api.main import create_app


def _make_geotiff_bytes(channels: int = 8, h: int = 64, w: int = 64) -> bytes:
    """Generate an in-memory GeoTIFF with random float32 data."""
    data = np.random.rand(channels, h, w).astype(np.float32)
    transform = from_bounds(west=-60, south=-10, east=-59, north=-9, width=w, height=h)
    buf = io.BytesIO()
    with rasterio.open(
        buf, "w",
        driver="GTiff",
        height=h, width=w,
        count=channels,
        dtype="float32",
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        dst.write(data)
    return buf.getvalue()


@pytest.fixture
def client():
    app = create_app()

    # Patch model to return zero logits (no detections)
    mock_model = MagicMock()
    import torch
    mock_model.return_value = torch.zeros(1, 1, 64, 64)
    mock_model.parameters = lambda: iter([torch.zeros(1)])
    mock_model.__call__ = lambda *a, **kw: torch.zeros(1, 1, 64, 64)

    with patch("forestguard.api.dependencies._load_model", return_value=mock_model):
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_health_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestAlerts:
    def test_list_alerts_empty(self, client):
        r = client.get("/alerts")
        assert r.status_code == 200
        data = r.json()
        assert "total" in data
        assert "alerts" in data

    def test_get_nonexistent_alert(self, client):
        r = client.get("/alerts/nonexistent-id-000")
        assert r.status_code == 404

    def test_export_geojson(self, client):
        r = client.get("/alerts/export")
        assert r.status_code == 200
        body = r.json()
        assert body["type"] == "FeatureCollection"
        assert "features" in body

    def test_list_alerts_severity_filter(self, client):
        r = client.get("/alerts?severity=high")
        assert r.status_code == 200
