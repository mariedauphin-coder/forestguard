
### Per-region pooled metrics, split = split_seed0

| variant | region | test tiles | positive px | n seeds | P (%) | R (%) | F1 (%) |
|---|---|---|---|---|---|---|---|
| early_only | Amazon | 17 | 7640 | 3 | 59.1 ± 7.8 | 26.2 ± 2.1 | 36.1 ± 1.1 |
| early_only | Carpathians | 17 | 1580 | 3 | 48.8 ± 3.7 | 39.6 ± 4.8 | 43.5 ± 1.6 |
| early_only | Congo | 15 | 7872 | 3 | 30.0 ± 2.5 | 53.0 ± 4.8 | 38.2 ± 1.7 |
| early_only | IvoryCoast | 15 | 4421 | 3 | 18.9 ± 17.8 | 1.1 ± 1.4 | 2.0 ± 2.6 |
| full | Amazon | 17 | 7640 | 5 | 61.5 ± 8.6 | 26.1 ± 2.4 | 36.4 ± 1.6 |
| full | Carpathians | 17 | 1580 | 5 | 41.4 ± 5.1 | 44.6 ± 3.2 | 42.6 ± 2.0 |
| full | Congo | 15 | 7872 | 5 | 26.0 ± 3.3 | 57.3 ± 4.6 | 35.6 ± 2.4 |
| full | IvoryCoast | 15 | 4421 | 5 | 15.3 ± 17.8 | 0.3 ± 0.3 | 0.5 ± 0.6 |
| late_only | Amazon | 17 | 7640 | 3 | 39.5 ± 17.4 | 14.3 ± 2.8 | 19.7 ± 2.1 |
| late_only | Carpathians | 17 | 1580 | 3 | 35.3 ± 7.6 | 43.3 ± 5.8 | 38.6 ± 6.3 |
| late_only | Congo | 15 | 7872 | 3 | 21.9 ± 1.1 | 69.8 ± 4.1 | 33.3 ± 0.8 |
| late_only | IvoryCoast | 15 | 4421 | 3 | 13.1 ± 8.0 | 1.0 ± 0.7 | 1.9 ± 1.3 |
| ndvi_diff | Amazon | 17 | 7640 | 1 | 5.2 | 20.6 | 8.3 |
| ndvi_diff | Carpathians | 17 | 1580 | 1 | 2.0 | 10.7 | 3.4 |
| ndvi_diff | Congo | 15 | 7872 | 1 | 21.2 | 45.9 | 29.0 |
| ndvi_diff | IvoryCoast | 15 | 4421 | 1 | 5.7 | 8.0 | 6.6 |
| no_cma | Amazon | 17 | 7640 | 3 | 64.2 ± 6.2 | 26.5 ± 2.5 | 37.3 ± 1.9 |
| no_cma | Carpathians | 17 | 1580 | 3 | 40.1 ± 0.7 | 42.9 ± 4.0 | 41.4 ± 1.5 |
| no_cma | Congo | 15 | 7872 | 3 | 25.5 ± 0.8 | 54.6 ± 7.1 | 34.7 ± 0.7 |
| no_cma | IvoryCoast | 15 | 4421 | 3 | 13.7 ± 23.8 | 1.2 ± 2.1 | 2.3 ± 3.9 |
| optical_only | Amazon | 17 | 7640 | 3 | 72.7 ± 6.7 | 23.3 ± 3.1 | 35.1 ± 2.9 |
| optical_only | Carpathians | 17 | 1580 | 3 | 42.1 ± 5.5 | 43.9 ± 3.2 | 42.7 ± 1.4 |
| optical_only | Congo | 15 | 7872 | 3 | 27.3 ± 1.9 | 53.5 ± 7.5 | 35.9 ± 1.0 |
| optical_only | IvoryCoast | 15 | 4421 | 3 | 18.3 ± 12.1 | 0.1 ± 0.1 | 0.2 ± 0.1 |
| sar_only | Amazon | 17 | 7640 | 3 | 17.5 ± 13.8 | 2.0 ± 1.9 | 3.5 ± 3.1 |
| sar_only | Carpathians | 17 | 1580 | 3 | 0.8 ± 0.7 | 0.5 ± 0.4 | 0.6 ± 0.5 |
| sar_only | Congo | 15 | 7872 | 3 | 16.8 ± 0.6 | 65.7 ± 3.0 | 26.8 ± 0.5 |
| sar_only | IvoryCoast | 15 | 4421 | 3 | 11.4 ± 3.5 | 3.3 ± 1.2 | 5.1 ± 1.8 |
| sar_ratio | Amazon | 17 | 7640 | 1 | 0.3 | 5.2 | 0.5 |
| sar_ratio | Carpathians | 17 | 1580 | 1 | 0.5 | 3.0 | 0.9 |
| sar_ratio | Congo | 15 | 7872 | 1 | 10.1 | 33.9 | 15.6 |
| sar_ratio | IvoryCoast | 15 | 4421 | 1 | 2.2 | 12.1 | 3.7 |
| unet_plain | Amazon | 17 | 7640 | 5 | 64.6 ± 3.0 | 27.4 ± 0.9 | 38.4 ± 0.9 |
| unet_plain | Carpathians | 17 | 1580 | 5 | 53.3 ± 4.4 | 42.7 ± 4.2 | 47.2 ± 3.3 |
| unet_plain | Congo | 15 | 7872 | 5 | 40.1 ± 2.4 | 49.8 ± 3.5 | 44.3 ± 0.6 |
| unet_plain | IvoryCoast | 15 | 4421 | 5 | 46.8 ± 13.0 | 2.1 ± 0.9 | 4.0 ± 1.5 |
| unet_r34 | Amazon | 17 | 7640 | 5 | 64.4 ± 2.2 | 25.7 ± 1.6 | 36.7 ± 1.4 |
| unet_r34 | Carpathians | 17 | 1580 | 5 | 53.3 ± 4.6 | 40.4 ± 3.2 | 45.7 ± 0.9 |
| unet_r34 | Congo | 15 | 7872 | 5 | 39.1 ± 2.4 | 48.9 ± 3.9 | 43.3 ± 0.5 |
| unet_r34 | IvoryCoast | 15 | 4421 | 5 | 43.0 ± 11.6 | 1.5 ± 0.8 | 2.8 ± 1.6 |

### Per-region pooled metrics, split = split_seed0_holdout_Congo

| variant | region | test tiles | positive px | n seeds | P (%) | R (%) | F1 (%) |
|---|---|---|---|---|---|---|---|
| full | Congo | 100 | 60170 | 3 | 19.2 ± 1.7 | 63.7 ± 4.1 | 29.5 ± 2.3 |
| unet_plain | Congo | 100 | 60170 | 3 | 24.6 ± 1.1 | 77.6 ± 3.0 | 37.3 ± 0.9 |
| unet_r34 | Congo | 100 | 60170 | 3 | 22.6 ± 1.1 | 80.3 ± 0.7 | 35.3 ± 1.4 |
