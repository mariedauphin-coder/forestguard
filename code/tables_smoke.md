
### Detection results, in-region test tiles (Table 2)

| Variant | n seeds | P (%) | R (%) | F1 (%) | IoU (%) | macro F1 (%) | macro IoU (%) | Params (M) | MACs (G) | model-only ms | end-to-end ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ndvi_diff [split_seed0.csv] | 1 | 72.8 | 83.4 | 77.8 | 63.6 | 75.5 | 70.3 | n/a | n/a | n/a | n/a |
| sar_ratio [split_seed0.csv] | 1 | 79.6 | 84.4 | 82.0 | 69.4 | 56.4 | 48.4 | n/a | n/a | n/a | n/a |
| unet_plain [split_seed0.csv] | 1 | 2.8 | 100.0 | 5.5 | 2.8 | 5.4 | 2.8 | 1.953185 | n/a | n/a | n/a |
| unet_r34 [split_seed0.csv] | 1 | 99.0 | 36.0 | 52.8 | 35.9 | 59.9 | 52.4 | 24.483 | 0.514 | 24.2 ± 0.3 | 25.1 ± 0.2 |
| full [split_seed0.csv] | 1 | 2.8 | 100.0 | 5.5 | 2.8 | 5.4 | 2.8 | 0.57 | 0.048 | 10.9 ± 0.4 | 11.8 ± 0.2 |

### Ablation (Table 4)

| Variant | n seeds | P (%) | R (%) | F1 (%) | IoU (%) | macro F1 (%) | macro IoU (%) | Params (M) | MACs (G) | model-only ms | end-to-end ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| optical_only [split_seed0.csv] | 1 | 2.8 | 100.0 | 5.5 | 2.8 | 5.4 | 2.8 | 0.569041 | n/a | n/a | n/a |
| sar_only [split_seed0.csv] | 1 | 2.8 | 100.0 | 5.5 | 2.8 | 5.4 | 2.8 | 0.567055 | n/a | n/a | n/a |
| early_only [split_seed0.csv] | 1 | 2.8 | 100.0 | 5.5 | 2.8 | 5.4 | 2.8 | 0.563289 | n/a | n/a | n/a |
| late_only [split_seed0.csv] | 1 | 0.0 | 0.0 | 0.0 | 0.0 | 32.1 | 32.1 | 0.015015 | n/a | n/a | n/a |
| no_cma [split_seed0.csv] | 1 | 2.8 | 100.0 | 5.5 | 2.8 | 5.4 | 2.8 | 0.567815 | n/a | n/a | n/a |
| full [split_seed0.csv] | 1 | 2.8 | 100.0 | 5.5 | 2.8 | 5.4 | 2.8 | 0.57 | 0.048 | 10.9 ± 0.4 | 11.8 ± 0.2 |

### Held-out region transfer (Table 3, last row)

| Variant | n seeds | P (%) | R (%) | F1 (%) | IoU (%) | macro F1 (%) | macro IoU (%) | Params (M) | MACs (G) | model-only ms | end-to-end ms |
|---|---|---|---|---|---|---|---|---|---|---|---|

### Per-region pooled metrics, ForestGuard full, in-region test tiles (Table 3)

| Region | P (%) | R (%) | F1 (%) | IoU (%) | seeds |
|---|---|---|---|---|---|
| Amazon | 2.4 | 100.0 | 4.6 | 2.4 | 1 |
| Carpathians | 3.9 | 100.0 | 7.5 | 3.9 | 1 |
| Congo | 3.0 | 100.0 | 5.9 | 3.0 | 1 |
| IvoryCoast | 1.5 | 100.0 | 3.0 | 1.5 | 1 |