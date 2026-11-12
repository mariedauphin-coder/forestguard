#!/bin/bash
# 5-minute check that the whole pipeline runs. SYNTHETIC data: ignore the numbers.
set -e
python synthetic.py --out synth --tiles 24 --size 64
python split.py --root synth --seed 0
python split.py --root synth --seed 0 --holdout_region Congo
for v in full no_cma early_only late_only optical_only sar_only unet_r34 unet_plain; do
  python train.py --root synth --split split_seed0.csv --variant $v --seed 0 --epochs 2 --bs 8 --workers 0 --out runs_smoke
done
python classical.py --root synth --split split_seed0.csv --out runs_smoke
python benchmark.py --variants full unet_r34 --reps 5 --warmup 2 --size 64 --out bench_smoke.json
python aggregate.py --runs runs_smoke --bench bench_smoke.json --out tables_smoke.md
