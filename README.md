# ForestGuard

Real-time deforestation detection using Sentinel-1 SAR + Sentinel-2 optical fusion with a Siamese MobileNetV2 + U-Net architecture.

## Overview

ForestGuard ingests paired before/after satellite scenes, fuses SAR and optical data at the pixel level, and produces georeferenced deforestation alerts with severity classifications. The system is designed for operational deployment via a FastAPI endpoint.

```
Sentinel-1 (VV/VH) ─┐
                      ├─► Fusion (8ch) ─► Siamese MobileNetV2 ─► U-Net Decoder ─► Alert GeoJSON
Sentinel-2 (6 bands) ─┘         before ──┘                  └── after
```

## Architecture

| Component | Details |
|---|---|
| **Encoder** | Shared-weight Siamese MobileNetV2 — processes before/after scenes independently, concatenates features at 4 scales |
| **Decoder** | U-Net with progressive upsampling and skip connections back to full input resolution |
| **Loss** | Weighted BCE + Dice (configurable `pos_weight` for class imbalance) |
| **Input** | 8-channel fused raster: S1 VV, S1 VH, S2 B2/B3/B4/B8/B11/B12 |
| **Output** | Binary deforestation mask → morphologically filtered → vectorised GeoJSON alerts |

## Project Structure

```
forestguard/
├── forestguard/
│   ├── preprocessing/      # Sentinel-1 SAR + Sentinel-2 optical preprocessing & fusion
│   ├── models/             # Siamese MobileNetV2 encoder, U-Net decoder, loss functions
│   ├── training/           # Dataset, trainer (AMP + cosine LR), metrics (IoU, F1)
│   ├── postprocessing/     # Morphological filtering, polygon vectorisation, alert generation
│   └── api/                # FastAPI app, Pydantic schemas, detection + alerts routers
├── scripts/
│   ├── preprocess.py       # Raw scenes → fused patches + train/val/test manifests
│   ├── train.py            # Model training
│   └── infer.py            # Inference → alerts.geojson
├── config/config.yaml      # All hyperparameters and pipeline settings
└── tests/                  # 20 unit tests (preprocessing, model, API)
```

## Quickstart

### 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Preprocess

```bash
python scripts/preprocess.py \
    --sar-before  data/raw/sentinel1/before/ \
    --opt-before  data/raw/sentinel2/before/ \
    --sar-after   data/raw/sentinel1/after/ \
    --opt-after   data/raw/sentinel2/after/ \
    --label       data/raw/labels/deforestation.tif \
    --out-dir     data/patches/
```

### 3. Train

```bash
python scripts/train.py \
    --train-manifest data/patches/train.json \
    --val-manifest   data/patches/val.json \
    --epochs 100 \
    --batch-size 16 \
    --device cuda
```

### 4. Infer

```bash
python scripts/infer.py \
    --before     data/processed/before_fused.tif \
    --after      data/processed/after_fused.tif \
    --checkpoint checkpoints/best.pt \
    --output     alerts.geojson
```

### 5. Serve

```bash
uvicorn forestguard.api.main:app --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/detect` | Upload before/after GeoTIFF pair → run inference → return alerts |
| `GET` | `/alerts` | List alerts (filter by severity, area, scene ID) |
| `GET` | `/alerts/{id}` | Fetch a single alert |
| `GET` | `/alerts/export` | Download all alerts as a GeoJSON FeatureCollection |
| `GET` | `/health` | Service health check |

## Configuration

All pipeline parameters are in `config/config.yaml`:

```yaml
sentinel1:
  polarizations: [VV, VH]
  clip_min: -25.0          # dB range for normalisation
  clip_max: 0.0

training:
  epochs: 100
  batch_size: 16
  pos_weight: 3.0          # upweight deforested class
  dice_weight: 0.5         # BCE/Dice balance

postprocessing:
  min_area_ha: 1.0         # ignore alerts smaller than 1 ha
  confidence_threshold: 0.5
```

## Tests

```bash
pytest tests/ -v
```

All 20 tests cover preprocessing transforms, model output shapes, loss behaviour, and API endpoints.

## Data Sources

- **Sentinel-1 GRD** — C-band SAR, IW mode, VV+VH polarisation, 10 m resolution
- **Sentinel-2 L2A** — Bottom-of-atmosphere reflectance, 10–20 m bands resampled to 10 m

Data can be downloaded via [Copernicus Data Space](https://dataspace.copernicus.eu/) or the `sentinelsat` library (included in requirements).
