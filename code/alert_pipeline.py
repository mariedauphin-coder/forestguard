"""
pipeline.py — Turning a blurry map into a message a park ranger can act on.

WHAT THIS DOES, IN KID LANGUAGE
-------------------------------
The model gives us a picture where every pixel is a number between 0 and 1,
like a heat map. That is useless to a forest ranger. A ranger needs:

    "Go to THIS spot. It is THIS big. It is THIS urgent."

So this file does five clean-up steps:

  STEP 1  DECIDE      Anything above 0.5 -> "yes, cleared". Below -> "no".
                      Now we have a black-and-white picture.

  STEP 2  GROUP       Find blobs of white pixels that touch each other.
                      Each blob = one clearing event.
                      Throw away blobs smaller than 1 hectare (probably
                      just noise, or one farmer's vegetable patch).

  STEP 3  MEASURE     For each blob work out:
                        - where is its middle? (GPS latitude/longitude)
                        - how big is it? (hectares)

  STEP 4  RANK        Small  (<5 ha)   -> Low       "write it down"
                      Medium (5-50 ha) -> Moderate  "tell the local officer"
                      Big    (>50 ha)  -> Critical  "call the government NOW"

  STEP 5  LABEL       Guess WHY the trees went:
                      farming / logging / mining / road-building

Then we hand over a tidy JSON message.
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional

import numpy as np
from scipy import ndimage


# Sentinel pixels are 10 m x 10 m = 100 m². One hectare = 10,000 m².
# So one hectare = 100 pixels.
PIXELS_PER_HECTARE = 100
MIN_EVENT_HECTARES = 1.0
THRESHOLD = 0.5

SEVERITY_BANDS = [
    (0.0,  5.0,  "Low",      "Log for monitoring. No dispatch."),
    (5.0,  50.0, "Moderate", "Notify regional monitoring officer."),
    (50.0, 1e9,  "Critical", "Escalate to enforcement immediately."),
]


@dataclass
class DeforestationAlert:
    """One clearing event, ready to send to a dashboard."""
    alert_id: str
    latitude: float
    longitude: float
    area_hectares: float
    severity: str
    recommended_action: str
    land_use_type: str
    confidence: float
    detected_at: str
    bbox: List[float]          # [min_lon, min_lat, max_lon, max_lat]

    def to_dict(self):
        return asdict(self)


def pixel_to_latlon(row: float, col: float, transform) -> tuple:
    """
    Convert "row 412, column 88 of the image" into real GPS coordinates.

    `transform` is the affine transform that comes with every GeoTIFF.
    It is basically a recipe: "pixel (0,0) is at this corner, and each
    pixel step moves you this far east and this far south."
    """
    lon, lat = transform * (col, row)
    return float(lat), float(lon)


def classify_severity(hectares: float) -> tuple:
    """Put the event into the Low / Moderate / Critical bucket."""
    for low, high, name, action in SEVERITY_BANDS:
        if low <= hectares < high:
            return name, action
    return "Critical", SEVERITY_BANDS[-1][3]


def classify_land_use(patch_rgb: np.ndarray) -> str:
    """
    A quick guess at WHY the forest disappeared.

    In the full system this is a small 3-layer CNN. Here is the simple
    rule-of-thumb version so you can see the logic clearly:

      straight edges + bare brown soil   -> Agricultural conversion
      ragged edges, some trees remaining -> Logging
      very bright bare ground + water    -> Mining
      long thin stripe                   -> Infrastructure (road)

    IMPORTANT: this is only a HINT. The paper reports 76.4% accuracy,
    which is good enough to help a ranger prioritise, but NOT good enough
    to use as evidence in court. A human must verify.
    """
    if patch_rgb is None or patch_rgb.size == 0:
        return "Unclassified"

    brightness = float(patch_rgb.mean())
    h, w = patch_rgb.shape[:2]
    elongation = max(h, w) / max(min(h, w), 1)

    if elongation > 6:
        return "Infrastructure"
    if brightness > 0.55:
        return "Mining"
    if brightness > 0.35:
        return "Agricultural conversion"
    return "Logging"


def generate_alerts(
    probability_map: np.ndarray,
    transform,
    optical_after: Optional[np.ndarray] = None,
    acquisition_date: Optional[datetime] = None,
    threshold: float = THRESHOLD,
) -> List[DeforestationAlert]:
    """
    The full five-step pipeline.

    probability_map : (H, W) float array, values 0..1, straight from the model
    transform       : rasterio affine transform, for GPS conversion
    optical_after   : (H, W, 3) RGB of the AFTER image, for the land-use guess
    acquisition_date: when the satellite took the AFTER picture

    Returns a list of alerts, biggest and most urgent first.
    """
    date = acquisition_date or datetime.utcnow()

    # STEP 1 — decide: turn the heat map into black and white
    binary = probability_map >= threshold

    # STEP 2 — group: find connected blobs of white pixels
    labelled, n_blobs = ndimage.label(binary)

    alerts: List[DeforestationAlert] = []

    for blob_id in range(1, n_blobs + 1):
        mask = labelled == blob_id
        pixel_count = int(mask.sum())

        # throw away anything smaller than 1 hectare
        hectares = pixel_count / PIXELS_PER_HECTARE
        if hectares < MIN_EVENT_HECTARES:
            continue

        # STEP 3 — measure: middle point and bounding box
        rows, cols = np.where(mask)
        centre_row, centre_col = rows.mean(), cols.mean()
        lat, lon = pixel_to_latlon(centre_row, centre_col, transform)

        lat_min, lon_min = pixel_to_latlon(rows.max(), cols.min(), transform)
        lat_max, lon_max = pixel_to_latlon(rows.min(), cols.max(), transform)

        # how sure was the model, on average, inside this blob?
        confidence = float(probability_map[mask].mean())

        # STEP 4 — rank
        severity, action = classify_severity(hectares)

        # STEP 5 — label
        land_use = "Unclassified"
        if optical_after is not None:
            patch = optical_after[rows.min():rows.max() + 1,
                                  cols.min():cols.max() + 1]
            land_use = classify_land_use(patch)

        alerts.append(DeforestationAlert(
            alert_id=f"FG-{date:%Y%m%d}-{blob_id:04d}",
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            area_hectares=round(hectares, 2),
            severity=severity,
            recommended_action=action,
            land_use_type=land_use,
            confidence=round(confidence, 3),
            detected_at=date.isoformat(),
            bbox=[round(lon_min, 6), round(lat_min, 6),
                  round(lon_max, 6), round(lat_max, 6)],
        ))

    # Most urgent first, then biggest first
    order = {"Critical": 0, "Moderate": 1, "Low": 2}
    alerts.sort(key=lambda a: (order[a.severity], -a.area_hectares))
    return alerts


if __name__ == "__main__":
    # Tiny demo you can run without any satellite data at all.
    from affine import Affine

    # Pretend the model found two clearings in a 500x500 image
    fake_map = np.zeros((500, 500), dtype=np.float32)
    fake_map[100:180, 100:200] = 0.93   # 80x100 px = 80 ha -> Critical
    fake_map[300:320, 300:330] = 0.71   # 20x30 px = 6 ha  -> Moderate

    # Pretend the top-left corner is at 2.5°S, 60.0°W, 10 m pixels
    transform = Affine(0.0001, 0, -60.0, 0, -0.0001, -2.5)

    for a in generate_alerts(fake_map, transform):
        print(f"[{a.severity:8}] {a.area_hectares:6.1f} ha  "
              f"@ {a.latitude:.4f}, {a.longitude:.4f}  -> {a.recommended_action}")
