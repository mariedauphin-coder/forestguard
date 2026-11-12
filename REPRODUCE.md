# Reproducing the paper

All commands are run from `code/`. Expected total time: GEE export ~1 h, 39 GPU runs ≈ 5 h on a Colab T4, CPU benchmark 10 min.

## 0. Environment
```bash
python3 -m venv fgenv && source fgenv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu   # CPU machine
pip install -r ../requirements-cpu.txt
```
Training was done on Colab with `torch 2.14.1+cu130`; the notebook installs nothing extra.

## 1. Data export (Earth Engine)
```bash
earthengine authenticate
export EE_PROJECT=your-cloud-project
python tools/gee_export.py --out raw_tiles --per_region 100     # 400 tiles, ~40 min
python label_check.py raw_tiles > label_check_400.txt   # sanity check of labels vs. NDVI/VH change
```
Or download the archived tiles from Zenodo (DOI in README) and extract to `raw_tiles/`.

## 2. Split
```bash
python split.py --root raw_tiles --seed 0                       # -> raw_tiles/split_seed0.csv  (228/65/64)
python split.py --root raw_tiles --seed 0 --holdout_region Congo # -> raw_tiles/split_seed0_holdout_Congo.csv
```

## 3. Training (GPU)
Open `notebooks/run_colab_v3.ipynb`, upload `raw_tiles` as `forestguard_data.tar` and the code as `fg_code.tar.gz` to Drive, run all cells.
Equivalent command for one run:
```bash
python train.py --root raw_tiles --split split_seed0.csv --variant full --seed 0 \
                --epochs 40 --bs 16 --lr 1e-3 --prior_bias -5 --out runs
```
Variants: `full no_cma early_only late_only optical_only sar_only unet_plain unet_r34`.
Seeds: 0–4 for `full`, `unet_plain`, `unet_r34`; 0–2 for the ablations and for the held-out split.

## 4. Classical baselines, tables, alerts (CPU)
```bash
python classical.py --root raw_tiles --split split_seed0.csv
python aggregate.py            # -> tables.md
python tools/per_region.py     # -> per_region.md
python tools/alerts.py         # -> alerts.md
```

## 5. CPU benchmark
Quiet machine, charger plugged in, nothing else running:
```bash
python benchmark.py --threads 4 --out benchmark_run1.json
```
Repeated on three separate occasions (`benchmark_run1..3.json`).

## Checks that a reproduction should pass
- `label_check_400.txt`: NDVI drop inside labelled loss is 0.10–0.18 per region, ~0 outside.
- `split_seed0.csv`: Amazon 54/19/17, Carpathians 51/16/17, Congo 64/15/15, IvoryCoast 59/15/15 (train/val/test).
- `tables.md`: `full` pooled test F1 within ~2 points of 32.7; `unet_plain` within ~1 point of 37.6.
