#!/usr/bin/env python3
"""
Run inference on a before/after fused raster pair and write alerts to GeoJSON.

Usage:
    python scripts/infer.py \
        --before      data/processed/before_fused.tif \
        --after       data/processed/after_fused.tif \
        --checkpoint  checkpoints/best.pt \
        --output      alerts.geojson \
        --threshold   0.5 \
        --min-area    1.0
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
import torch

from forestguard.models import ForestGuardModel
from forestguard.postprocessing import AlertGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ForestGuard inference")
    parser.add_argument("--before",      required=True)
    parser.add_argument("--after",       required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--output",      default="alerts.geojson")
    parser.add_argument("--threshold",   type=float, default=0.5)
    parser.add_argument("--min-area",    type=float, default=1.0)
    parser.add_argument("--scene-id",    default="")
    parser.add_argument("--device",      default="cuda")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    logger.info("Loading model from %s", args.checkpoint)
    model = ForestGuardModel.from_checkpoint(args.checkpoint)
    model.eval().to(device)

    logger.info("Loading rasters…")
    with rasterio.open(args.before) as src:
        before_arr = src.read().astype(np.float32)
        transform = src.transform
        crs = src.crs

    with rasterio.open(args.after) as src:
        after_arr = src.read().astype(np.float32)

    h = min(before_arr.shape[1], after_arr.shape[1])
    w = min(before_arr.shape[2], after_arr.shape[2])
    before_t = torch.from_numpy(before_arr[:, :h, :w]).unsqueeze(0).to(device)
    after_t  = torch.from_numpy(after_arr[:, :h, :w]).unsqueeze(0).to(device)

    logger.info("Running inference…")
    with torch.no_grad():
        logits = model(before_t, after_t).squeeze(0)

    alert_gen = AlertGenerator(
        confidence_threshold=args.threshold,
        min_area_ha=args.min_area,
    )
    alerts = alert_gen.generate(
        logits, transform, crs,
        scene_date=datetime.now(timezone.utc),
        scene_id=args.scene_id or Path(args.after).stem,
    )

    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": a["geometry"],
                "properties": {k: v for k, v in a.items() if k != "geometry"},
            }
            for a in alerts
        ],
    }

    with open(args.output, "w") as f:
        json.dump(geojson, f, indent=2, default=str)

    logger.info("Wrote %d alerts to %s", len(alerts), args.output)


if __name__ == "__main__":
    main()
