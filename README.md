# ForestGuard — SAR–optical fusion for annual deforestation mapping

Code, splits, run logs and result tables for the paper
*"ForestGuard: A Lightweight SAR–Optical Fusion Framework for Near-Real-Time Deforestation Alerts for NGOs and Governments"* (IJIBC, revised submission).

Every number in the paper is in `results/` and was produced by the code in `code/` on the data described below.
Nothing was rounded up, no seed was dropped, no run was repeated to look better. The append-only
`docs/LAB_NOTEBOOK.md` records every decision, including mistakes and their corrections.

## What is here

| Folder | Contents |
|---|---|
| `code/` | the exact scripts that were run: Earth Engine export, split, training, classical baselines, CPU benchmark, aggregation |
| `results/tables.md` | pooled/macro test metrics, mean ± SD over seeds (the source of Table 2–3 in the paper) |
| `results/per_region.md` | in-region test metrics (Table 4) |
| `results/alerts.md` | tile-level alert metrics at K = 100 px and 500 px, with and without Ivory Coast (Table 5) |
| `results/benchmark*.json` | CPU latency runs (Dell Inspiron 15-3530, 4 threads, fp32, batch 1) |
| `results/runs/<run>/result.json` | per-run, per-seed test metrics and best validation epoch |
| `results/split*.csv`, `results/tile_manifest.csv` | the train/val/test assignment of every tile and each tile's UTM origin |
| `notebooks/run_colab_v3.ipynb` | the Colab notebook that ran all 39 GPU runs |
| `docs/LAB_NOTEBOOK.md` | append-only lab notebook |

## Data (not in this repo — too large)

400 tiles of 256×256 px at 10 m (2.56 km squares) on UTM grids, 100 per region:

| Region | Box (lon/lat) | Season | CRS | 2022 loss px |
|---|---|---|---|---|
| Amazon (Rondônia) | [-63.2, -9.8, -62.6, -9.3] | Jun–Sep | EPSG:32720 | 40,052 |
| Congo Basin | [23.5, -1.0, 24.1, -0.5] | Jun–Aug | EPSG:32734 | 60,170 |
| Ivory Coast | [-6.6, 5.9, -6.0, 6.4] | Jan–Mar | EPSG:32629 | 29,749 |
| Carpathians (Bucovina) | [25.3, 47.3, 25.9, 47.7] | Jun–Sep | EPSG:32635 | 23,066 |

Inputs: Sentinel-1 GRD (VV, VH, dB) and Sentinel-2 SR Harmonized (B2, B3, B4, B5, B8, B11, NDVI, NDMI; SCL cloud mask, <40 % cloudy scenes), median composites of the same calendar window in 2021 (pre) and 2022 (post).
Labels: Hansen Global Forest Change v1.13, `lossyear == 22` and `treecover2000 > 30`.
The tiles (`.npz` + `manifest.csv`) and the trained weights (`model.pt`, 39 runs) are archived on Zenodo: **DOI to be added**.
`code/tools/gee_export.py` regenerates the tiles from Earth Engine (set `EE_PROJECT` to your own cloud project).

## Protocol in one paragraph

Block split (4×4-tile blocks, 15 % val, 15 % test per region, 1-tile buffer, seed 0): 228 / 65 / 64 tiles.
One recipe for every model, chosen on validation with the `full` variant (seed 0, two trials) and then frozen:
Adam lr 1e-3, cosine annealing, 40 epochs, batch 16, output bias initialised to −5, loss 0.4·BCE + 0.6·Dice on valid pixels,
flip/rot90 augmentation, no pretraining, best epoch by validation pooled F1 at threshold 0.5, test evaluated once.
ForestGuard variants: `full` ×5 seeds, ablations ×3 seeds; baselines `unet_plain`, `unet_r34` ×5 seeds;
held-out-Congo split ×3 seeds. Classical baselines `ndvi_diff` and `sar_ratio` with thresholds tuned on validation tiles.

## Headline results (test, pooled, %, mean ± SD)

| Model | Params | F1 | IoU | CPU ms (run 1) |
|---|---|---|---|---|
| ndvi_diff | – | 13.7 | 7.3 | – |
| sar_ratio | – | 3.3 | 1.7 | – |
| ForestGuard full | 570,119 | 32.7 ± 1.2 | 19.5 ± 0.9 | 36.7 ± 3.4 |
| U-Net (plain) | 1.953 M | **37.6 ± 0.5** | **23.1 ± 0.4** | 183.1 ± 11.2 |
| U-Net (ResNet-34) | 24.483 M | 36.4 ± 0.5 | 22.2 ± 0.4 | 140.2 ± 13.1 |

Exact parameter and MAC counts are in `results/benchmark_run1.json`.
ForestGuard reaches about 87 % of the best F1 at 3.4× fewer parameters and ~5× lower CPU latency than the plain U-Net.
It does **not** beat the U-Net baselines. SAR-only is far below optical-only (19.8 vs 32.4), and fusion gives no measurable
gain over optical-only in these clear-sky composites. Every method fails on the Ivory Coast tiles (F1 ≤ 6.6);
see `results/alerts.md` and the lab notebook for the diagnosis.

See `REPRODUCE.md` to rerun everything.

## Citation / licence

Code: MIT (see `LICENSE`). Data: Sentinel-1/2 © ESA Copernicus; labels © Hansen/UMD/Google/USGS/NASA (CC-BY 4.0).
