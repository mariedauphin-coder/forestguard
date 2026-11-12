
### Pooled pixel metrics, ALL regions (should equal aggregate.py)

| split | variant | n seeds | P | R | F1 | IoU |
|---|---|---|---|---|---|---|
| split_seed0 | early_only | 3 | 36.3 ± 2.3 | 31.8 ± 1.7 | 33.8 ± 1.0 | 20.4 ± 0.7 |
| split_seed0 | full | 5 | 32.2 ± 3.7 | 33.6 ± 2.6 | 32.7 ± 1.2 | 19.5 ± 0.9 |
| split_seed0 | late_only | 3 | 23.9 ± 2.7 | 34.0 ± 2.0 | 28.0 ± 1.3 | 16.3 ± 0.9 |
| split_seed0 | ndvi_diff | 1 | 9.2 | 26.5 | 13.7 | 7.3 |
| split_seed0 | no_cma | 3 | 32.4 ± 1.6 | 32.8 ± 4.1 | 32.4 ± 1.3 | 19.4 ± 1.0 |
| split_seed0 | optical_only | 3 | 34.1 ± 2.1 | 31.1 ± 3.4 | 32.4 ± 0.8 | 19.3 ± 0.6 |
| split_seed0 | sar_only | 3 | 16.2 ± 0.3 | 25.5 ± 1.6 | 19.8 ± 0.3 | 11.0 ± 0.2 |
| split_seed0 | sar_ratio | 1 | 1.8 | 17.0 | 3.3 | 1.7 |
| split_seed0 | unet_plain | 5 | 46.7 ± 2.5 | 31.5 ± 1.5 | 37.6 ± 0.5 | 23.1 ± 0.4 |
| split_seed0 | unet_r34 | 5 | 45.7 ± 2.6 | 30.3 ± 1.6 | 36.4 ± 0.5 | 22.2 ± 0.4 |
| split_seed0_holdout_Congo | full | 3 | 19.2 ± 1.7 | 63.7 ± 4.1 | 29.5 ± 2.3 | 17.3 ± 1.6 |
| split_seed0_holdout_Congo | unet_plain | 3 | 24.6 ± 1.1 | 77.6 ± 3.0 | 37.3 ± 0.9 | 23.0 ± 0.7 |
| split_seed0_holdout_Congo | unet_r34 | 3 | 22.6 ± 1.1 | 80.3 ± 0.7 | 35.3 ± 1.4 | 21.4 ± 1.0 |

### Pooled pixel metrics EXCLUDING IvoryCoast (post-hoc, no retraining)

| split | variant | n seeds | P | R | F1 | IoU |
|---|---|---|---|---|---|---|
| split_seed0 | early_only | 3 | 36.4 ± 2.3 | 39.8 ± 1.9 | 37.9 ± 0.9 | 23.4 ± 0.7 |
| split_seed0 | full | 5 | 32.3 ± 3.7 | 42.2 ± 3.2 | 36.3 ± 1.6 | 22.2 ± 1.2 |
| split_seed0 | late_only | 3 | 24.0 ± 2.7 | 42.5 ± 2.5 | 30.6 ± 1.7 | 18.1 ± 1.2 |
| split_seed0 | ndvi_diff | 1 | 9.6 | 31.3 | 14.7 | 7.9 |
| split_seed0 | no_cma | 3 | 32.3 ± 1.6 | 41.0 ± 4.6 | 36.0 ± 0.9 | 21.9 ± 0.7 |
| split_seed0 | optical_only | 3 | 34.1 ± 2.1 | 39.1 ± 4.2 | 36.3 ± 0.6 | 22.1 ± 0.4 |
| split_seed0 | sar_only | 3 | 16.4 ± 0.1 | 31.2 ± 2.2 | 21.5 ± 0.4 | 12.0 ± 0.3 |
| split_seed0 | sar_ratio | 1 | 1.7 | 18.2 | 3.2 | 1.6 |
| split_seed0 | unet_plain | 5 | 46.8 ± 2.7 | 39.1 ± 2.0 | 42.5 ± 0.4 | 27.0 ± 0.3 |
| split_seed0 | unet_r34 | 5 | 45.8 ± 2.6 | 37.7 ± 2.1 | 41.3 ± 0.4 | 26.0 ± 0.3 |
| split_seed0_holdout_Congo | full | 3 | 19.2 ± 1.7 | 63.7 ± 4.1 | 29.5 ± 2.3 | 17.3 ± 1.6 |
| split_seed0_holdout_Congo | unet_plain | 3 | 24.6 ± 1.1 | 77.6 ± 3.0 | 37.3 ± 0.9 | 23.0 ± 0.7 |
| split_seed0_holdout_Congo | unet_r34 | 3 | 22.6 ± 1.1 | 80.3 ± 0.7 | 35.3 ± 1.4 | 21.4 ± 1.0 |

### Tile alerts, K = 100 px (alert if predicted/true loss px in the tile >= K), ALL regions

| split | variant | n seeds | P | R | FDP | F1 | true_alert_tiles |
|---|---|---|---|---|---|---|---|
| split_seed0 | early_only | 3 | 92.6 ± 2.7 | 49.5 ± 3.5 | 7.4 ± 2.7 | 64.4 ± 2.8 | 33.0 ± 0.0 |
| split_seed0 | full | 5 | 91.6 ± 5.8 | 50.3 ± 1.7 | 8.4 ± 5.8 | 64.9 ± 1.5 | 33.0 ± 0.0 |
| split_seed0 | late_only | 3 | 81.8 ± 8.5 | 48.5 ± 3.0 | 18.2 ± 8.5 | 60.8 ± 4.0 | 33.0 ± 0.0 |
| split_seed0 | ndvi_diff | 1 | 58.3 | 84.8 | 41.7 | 69.1 | 33.0 |
| split_seed0 | no_cma | 3 | 91.9 ± 4.3 | 54.5 ± 5.2 | 8.1 ± 4.3 | 68.3 ± 2.7 | 33.0 ± 0.0 |
| split_seed0 | optical_only | 3 | 89.4 ± 4.7 | 50.5 ± 1.7 | 10.6 ± 4.7 | 64.5 ± 2.0 | 33.0 ± 0.0 |
| split_seed0 | sar_only | 3 | 74.4 ± 10.0 | 45.5 ± 3.0 | 25.6 ± 10.0 | 56.3 ± 4.7 | 33.0 ± 0.0 |
| split_seed0 | sar_ratio | 1 | 54.1 | 100.0 | 45.9 | 70.2 | 33.0 |
| split_seed0 | unet_plain | 5 | 88.3 ± 1.9 | 50.3 ± 1.7 | 11.7 ± 1.9 | 64.1 ± 1.3 | 33.0 ± 0.0 |
| split_seed0 | unet_r34 | 5 | 88.9 ± 0.6 | 48.5 ± 3.0 | 11.1 ± 0.6 | 62.7 ± 2.7 | 33.0 ± 0.0 |
| split_seed0_holdout_Congo | full | 3 | 85.3 ± 7.0 | 95.3 ± 1.0 | 14.7 ± 7.0 | 89.9 ± 3.5 | 57.0 ± 0.0 |
| split_seed0_holdout_Congo | unet_plain | 3 | 96.4 ± 0.0 | 94.7 ± 0.0 | 3.6 ± 0.0 | 95.6 ± 0.0 | 57.0 ± 0.0 |
| split_seed0_holdout_Congo | unet_r34 | 3 | 95.3 ± 1.9 | 94.7 ± 0.0 | 4.7 ± 1.9 | 95.0 ± 1.0 | 57.0 ± 0.0 |

### Tile alerts, K = 100 px, EXCLUDING IvoryCoast (post-hoc)

| split | variant | n seeds | P | R | FDP | F1 | true_alert_tiles |
|---|---|---|---|---|---|---|---|
| split_seed0 | early_only | 3 | 92.4 ± 2.6 | 72.7 ± 4.5 | 7.6 ± 2.6 | 81.3 ± 2.1 | 22.0 ± 0.0 |
| split_seed0 | full | 5 | 91.6 ± 5.8 | 75.5 ± 2.5 | 8.4 ± 5.8 | 82.6 ± 2.1 | 22.0 ± 0.0 |
| split_seed0 | late_only | 3 | 81.8 ± 8.5 | 72.7 ± 4.5 | 18.2 ± 8.5 | 76.9 ± 5.4 | 22.0 ± 0.0 |
| split_seed0 | ndvi_diff | 1 | 51.4 | 81.8 | 48.6 | 63.2 | 22.0 |
| split_seed0 | no_cma | 3 | 91.3 ± 5.5 | 77.3 ± 0.0 | 8.7 ± 5.5 | 83.7 ± 2.3 | 22.0 ± 0.0 |
| split_seed0 | optical_only | 3 | 89.4 ± 4.7 | 75.8 ± 2.6 | 10.6 ± 4.7 | 82.0 ± 2.7 | 22.0 ± 0.0 |
| split_seed0 | sar_only | 3 | 69.1 ± 12.9 | 53.0 ± 6.9 | 30.9 ± 12.9 | 60.0 ± 9.3 | 22.0 ± 0.0 |
| split_seed0 | sar_ratio | 1 | 47.8 | 100.0 | 52.2 | 64.7 | 22.0 |
| split_seed0 | unet_plain | 5 | 88.2 ± 1.8 | 74.5 ± 2.5 | 11.8 ± 1.8 | 80.8 ± 1.3 | 22.0 ± 0.0 |
| split_seed0 | unet_r34 | 5 | 88.9 ± 0.6 | 72.7 ± 4.5 | 11.1 ± 0.6 | 79.9 ± 3.0 | 22.0 ± 0.0 |
| split_seed0_holdout_Congo | full | 3 | 85.3 ± 7.0 | 95.3 ± 1.0 | 14.7 ± 7.0 | 89.9 ± 3.5 | 57.0 ± 0.0 |
| split_seed0_holdout_Congo | unet_plain | 3 | 96.4 ± 0.0 | 94.7 ± 0.0 | 3.6 ± 0.0 | 95.6 ± 0.0 | 57.0 ± 0.0 |
| split_seed0_holdout_Congo | unet_r34 | 3 | 95.3 ± 1.9 | 94.7 ± 0.0 | 4.7 ± 1.9 | 95.0 ± 1.0 | 57.0 ± 0.0 |

### Tile alerts, K = 500 px (alert if predicted/true loss px in the tile >= K), ALL regions

| split | variant | n seeds | P | R | FDP | F1 | true_alert_tiles |
|---|---|---|---|---|---|---|---|
| split_seed0 | early_only | 3 | 73.1 ± 3.3 | 64.3 ± 0.0 | 26.9 ± 3.3 | 68.4 ± 1.5 | 14.0 ± 0.0 |
| split_seed0 | full | 5 | 75.4 ± 0.9 | 65.7 ± 3.2 | 24.6 ± 0.9 | 70.2 ± 2.2 | 14.0 ± 0.0 |
| split_seed0 | late_only | 3 | 69.3 ± 4.9 | 59.5 ± 10.9 | 30.7 ± 4.9 | 63.8 ± 7.7 | 14.0 ± 0.0 |
| split_seed0 | ndvi_diff | 1 | 39.3 | 78.6 | 60.7 | 52.4 | 14.0 |
| split_seed0 | no_cma | 3 | 75.6 ± 1.1 | 66.7 ± 4.1 | 24.4 ± 1.1 | 70.8 ± 2.8 | 14.0 ± 0.0 |
| split_seed0 | optical_only | 3 | 74.2 ± 1.3 | 61.9 ± 4.1 | 25.8 ± 1.3 | 67.5 ± 3.0 | 14.0 ± 0.0 |
| split_seed0 | sar_only | 3 | 63.9 ± 4.8 | 45.2 ± 4.1 | 36.1 ± 4.8 | 52.7 ± 1.0 | 14.0 ± 0.0 |
| split_seed0 | sar_ratio | 1 | 24.1 | 92.9 | 75.9 | 38.2 | 14.0 |
| split_seed0 | unet_plain | 5 | 76.4 ± 3.0 | 64.3 ± 0.0 | 23.6 ± 3.0 | 69.8 ± 1.2 | 14.0 ± 0.0 |
| split_seed0 | unet_r34 | 5 | 75.0 ± 0.0 | 64.3 ± 0.0 | 25.0 ± 0.0 | 69.2 ± 0.0 | 14.0 ± 0.0 |
| split_seed0_holdout_Congo | full | 3 | 76.5 ± 4.2 | 100.0 ± 0.0 | 23.5 ± 4.2 | 86.6 ± 2.8 | 43.0 ± 0.0 |
| split_seed0_holdout_Congo | unet_plain | 3 | 82.7 ± 0.0 | 100.0 ± 0.0 | 17.3 ± 0.0 | 90.5 ± 0.0 | 43.0 ± 0.0 |
| split_seed0_holdout_Congo | unet_r34 | 3 | 81.2 ± 1.5 | 100.0 ± 0.0 | 18.8 ± 1.5 | 89.6 ± 0.9 | 43.0 ± 0.0 |

### Tile alerts, K = 500 px, EXCLUDING IvoryCoast (post-hoc)

| split | variant | n seeds | P | R | FDP | F1 | true_alert_tiles |
|---|---|---|---|---|---|---|---|
| split_seed0 | early_only | 3 | 73.1 ± 3.3 | 75.0 ± 0.0 | 26.9 ± 3.3 | 74.0 ± 1.7 | 12.0 ± 0.0 |
| split_seed0 | full | 5 | 75.4 ± 0.9 | 76.7 ± 3.7 | 24.6 ± 0.9 | 76.0 ± 2.2 | 12.0 ± 0.0 |
| split_seed0 | late_only | 3 | 69.3 ± 4.9 | 69.4 ± 12.7 | 30.7 ± 4.9 | 69.1 ± 8.0 | 12.0 ± 0.0 |
| split_seed0 | ndvi_diff | 1 | 40.9 | 75.0 | 59.1 | 52.9 | 12.0 |
| split_seed0 | no_cma | 3 | 75.6 ± 1.1 | 77.8 ± 4.8 | 24.4 ± 1.1 | 76.7 ± 2.9 | 12.0 ± 0.0 |
| split_seed0 | optical_only | 3 | 74.2 ± 1.3 | 72.2 ± 4.8 | 25.8 ± 1.3 | 73.2 ± 3.1 | 12.0 ± 0.0 |
| split_seed0 | sar_only | 3 | 62.6 ± 7.0 | 50.0 ± 0.0 | 37.4 ± 7.0 | 55.5 ± 2.9 | 12.0 ± 0.0 |
| split_seed0 | sar_ratio | 1 | 28.2 | 91.7 | 71.8 | 43.1 | 12.0 |
| split_seed0 | unet_plain | 5 | 76.4 ± 3.0 | 75.0 ± 0.0 | 23.6 ± 3.0 | 75.7 ± 1.5 | 12.0 ± 0.0 |
| split_seed0 | unet_r34 | 5 | 75.0 ± 0.0 | 75.0 ± 0.0 | 25.0 ± 0.0 | 75.0 ± 0.0 | 12.0 ± 0.0 |
| split_seed0_holdout_Congo | full | 3 | 76.5 ± 4.2 | 100.0 ± 0.0 | 23.5 ± 4.2 | 86.6 ± 2.8 | 43.0 ± 0.0 |
| split_seed0_holdout_Congo | unet_plain | 3 | 82.7 ± 0.0 | 100.0 ± 0.0 | 17.3 ± 0.0 | 90.5 ± 0.0 | 43.0 ± 0.0 |
| split_seed0_holdout_Congo | unet_r34 | 3 | 81.2 ± 1.5 | 100.0 ± 0.0 | 18.8 ± 1.5 | 89.6 ± 0.9 | 43.0 ± 0.0 |
