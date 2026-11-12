# ForestGuard experiments

Everything the reviewers asked to verify is produced by these scripts. **No number may enter the paper unless it came from a run of this code on real data.**

## Pipeline
1. **Data** – `python tools/gee_export.py --out raw_tiles` (Earth Engine, Hansen GFC labels) **or** write `manifest.csv` + `.npz` patches in the format described at the top of `data.py`.
2. **Split** – `python split.py --root raw_tiles --seed 0` (tile-level, buffered) and `python split.py --root raw_tiles --seed 0 --holdout_region Congo` (transfer test).
3. **Train** – `python train.py --root raw_tiles --split split_seed0.csv --variant full --seed 0` for each variant and each seed 0-4:
   `full no_cma early_only late_only optical_only sar_only unet_r34 unet_plain` (add `unet_r34_opt`).
4. **Classical baselines** – `python classical.py --root raw_tiles --split split_seed0.csv`.
5. **Benchmark (run on the Dell CPU, not Colab)** – `python benchmark.py --threads 4`.
6. **Tables** – `python aggregate.py` → `tables.md`.

`run_colab.ipynb` runs steps 2–4 on a free Colab GPU. `synthetic.py` makes fake data to test the code; its results are meaningless.

## Honesty notes
- GLAD-L, CuSum NRT and the transformer fusion model [Ferrari et al.] are **not re-implemented**. Either drop them from the baseline table or cite their published numbers separately, labelled "reported on a different dataset".
- Labels are Hansen GFC (annual), not GLAD-S2/PRODES. Update the paper text to match.
- The test tiles are evaluated once, with the epoch chosen on validation tiles; the threshold is 0.5 and is not tuned on test.
