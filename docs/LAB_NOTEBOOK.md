# Lab notebook (append-only; corrections are added, never erased)


## lab_notebook.txt
```
2026-11-12  Setup: Ubuntu 24.04, Dell Inspiron 15-3530, python venv fgenv, torch 2.15.0+cpu. smoke_test.sh passed on synthetic data.
2026-11-12  Data: Earth Engine export (tools/gee_export.py). Labels: Hansen GFC 2025_v1_13, loss in after-year 2022, treecover2000 > 30. Sentinel-1 GRD IW VV+VH median composite, Sentinel-2 SR median composite (SCL cloud mask).
2026-11-12  Fix: NaN pixels in the export are set to 0 and marked invalid.
2026-11-12  Label check on 5 tiles per region: NDVI drops and radar drops inside the label in Amazon and Congo. Ivory Coast failed with the Dec-Feb window.
2026-11-12  Decision: Ivory Coast window changed from Dec-Feb to Jan-Mar. Based on label/image agreement only, not on model results. After the change: NDVI drop inside 0.116 vs outside 0.030 (5 tiles).
2026-11-12  Running export: 30 tiles per region -> raw_tiles/ (log: export_log_30.txt)

```

## lab_scan.txt
```
Amazon (current)             forest= 73.1% of area   loss2022= 2.50% of area   (3.42% of forest)
Congo (current)              forest= 99.9% of area   loss2022= 5.07% of area   (5.08% of forest)
IvoryCoast (current)         forest= 79.8% of area   loss2022= 2.81% of area   (3.52% of forest)
Carpathians (current)        forest= 44.6% of area   loss2022= 4.20% of area   (9.43% of forest)
Bucovina                     forest= 71.1% of area   loss2022= 3.26% of area   (4.58% of forest)
Bistrita                     forest= 70.4% of area   loss2022= 3.22% of area   (4.58% of forest)
Harghita                     forest= 50.0% of area   loss2022= 3.14% of area   (6.28% of forest)
Vrancea                      forest= 62.1% of area   loss2022= 4.08% of area   (6.56% of forest)
Maramures                    forest= 66.1% of area   loss2022= 1.68% of area   (2.54% of forest)
Arges-Valcea                 forest= 69.7% of area   loss2022= 2.67% of area   (3.83% of forest)
Zakarpattia                  forest= 74.4% of area   loss2022= 2.17% of area   (2.91% of forest)
Amazon_0_16            EE@10m=  0.09%  EE@30m=  0.12%  saved=  0.05%
Amazon_0_3             EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Amazon_10_9            EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Amazon_11_10           EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Amazon_11_16           EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Amazon_12_1            EE@10m=  2.60%  EE@30m=  2.17%  saved=  0.64%
Amazon_12_5            EE@10m=  7.22%  EE@30m=  7.01%  saved=  2.73%
Amazon_14_13           EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Amazon_14_17           EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Amazon_15_0            EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Carpathians_0_1        EE@10m=  4.81%  EE@30m=  4.48%  saved=  0.05%
Carpathians_0_11       EE@10m=  3.96%  EE@30m=  3.31%  saved=  0.16%
Carpathians_0_13       EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Carpathians_10_12      EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Carpathians_10_14      EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Carpathians_10_5       EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Carpathians_11_14      EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Carpathians_11_4       EE@10m=  1.18%  EE@30m=  0.83%  saved=  0.00%
Carpathians_11_7       EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Carpathians_12_16      EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Congo_0_5              EE@10m=  3.67%  EE@30m=  3.49%  saved=  1.29%
Congo_10_11            EE@10m=  3.29%  EE@30m=  3.09%  saved=  1.41%
Congo_10_20            EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Congo_11_18            EE@10m=  8.48%  EE@30m=  8.14%  saved=  0.73%
Congo_12_17            EE@10m= 43.53%  EE@30m= 46.53%  saved=  0.56%
Congo_13_10            EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Congo_14_14            EE@10m=  2.99%  EE@30m=  3.06%  saved=  1.74%
Congo_17_14            EE@10m= 10.71%  EE@30m= 10.66%  saved=  4.06%
Congo_17_5             EE@10m=  0.00%  EE@30m=  0.00%  saved=  0.00%
Congo_18_12            EE@10m=  1.63%  EE@30m=  1.46%  saved=  0.37%
IvoryCoast_0_1         EE@10m=  1.45%  EE@30m=  1.31%  saved=  0.30%
IvoryCoast_0_7         EE@10m=  5.62%  EE@30m=  5.41%  saved=  0.76%
IvoryCoast_10_19       EE@10m=  2.55%  EE@30m=  2.31%  saved=  0.37%
IvoryCoast_11_0        EE@10m=  1.05%  EE@30m=  1.26%  saved=  0.11%
IvoryCoast_11_19       EE@10m=  0.45%  EE@30m=  0.46%  saved=  0.05%
IvoryCoast_12_13       EE@10m=  0.39%  EE@30m=  0.23%  saved=  0.04%
IvoryCoast_12_20       EE@10m=  1.47%  EE@30m=  1.72%  saved=  0.18%
IvoryCoast_13_19       EE@10m=  0.18%  EE@30m=  0.15%  saved=  0.02%
IvoryCoast_13_7        EE@10m=  0.62%  EE@30m=  0.75%  saved=  0.05%
IvoryCoast_13_9        EE@10m=  0.42%  EE@30m=  0.30%  saved=  0.07%

mean over the first 10 tiles per region:
Amazon       EE@10m=  0.99%  EE@30m=  0.93%  saved=  0.34%
Carpathians  EE@10m=  1.00%  EE@30m=  0.86%  saved=  0.02%
Congo        EE@10m=  7.43%  EE@30m=  7.64%  saved=  1.02%
IvoryCoast   EE@10m=  1.42%  EE@30m=  1.39%  saved=  0.19%

share of forest pixels lost, by year (whole box, 30 m):
Amazon       2018:3.90%  2019:4.24%  2020:2.94%  2021:3.28%  2022:2.67%  2023:1.57%  2024:0.67%
Congo        2018:7.00%  2019:6.55%  2020:5.86%  2021:5.43%  2022:5.08%  2023:5.16%  2024:4.14%
IvoryCoast   2018:3.32%  2019:8.98%  2020:3.52%  2021:3.38%  2022:3.12%  2023:2.91%  2024:4.47%
Carpathians  2018:2.44%  2019:3.21%  2020:5.13%  2021:4.00%  2022:4.28%  2023:5.88%  2024:6.01%
Congo_12_17: Earth Engine loss px @10m = 363;  saved npz loss px = 365
label only                 shape=(257, 257) bands=2  loss px full=365  top-left 256x256=365  NaN=0
label + S1 after           shape=(257, 257) bands=4  loss px full=365  top-left 256x256=365  NaN=0
label + S2 after           shape=(257, 257) bands=10  loss px full=365  top-left 256x256=365  NaN=0
full stack (as exported)   shape=(257, 257) bands=22  loss px full=365  top-left 256x256=365  NaN=0
check Congo_12_17: loss share of area = 0.55%  (expect about 0.55%)

Amazon (current)             forest= 73.1% of area   loss2022= 0.64% of area   (0.88% of forest)
Congo (current)              forest= 99.9% of area   loss2022= 0.80% of area   (0.80% of forest)
IvoryCoast (current)         forest= 79.8% of area   loss2022= 0.41% of area   (0.51% of forest)
Carpathians (current)        forest= 44.6% of area   loss2022= 0.10% of area   (0.23% of forest)
Bucovina                     forest= 71.1% of area   loss2022= 0.36% of area   (0.51% of forest)
Bistrita                     forest= 70.4% of area   loss2022= 0.15% of area   (0.21% of forest)
Harghita                     forest= 50.0% of area   loss2022= 0.25% of area   (0.50% of forest)
Vrancea                      forest= 62.1% of area   loss2022= 0.16% of area   (0.25% of forest)
Maramures                    forest= 66.1% of area   loss2022= 0.08% of area   (0.12% of forest)
Arges-Valcea                 forest= 69.7% of area   loss2022= 0.14% of area   (0.19% of forest)
Zakarpattia                  forest= 74.4% of area   loss2022= 0.07% of area   (0.10% of forest)

Carpathians tile array shape (rows, cols): (257, 367)
```

## lab_split.txt
```
{
 "Amazon": {
  "buffer_dropped": 10,
  "test": 17,
  "train": 54,
  "val": 19
 },
 "Carpathians": {
  "buffer_dropped": 16,
  "test": 17,
  "train": 51,
  "val": 16
 },
 "Congo": {
  "buffer_dropped": 6,
  "test": 15,
  "train": 64,
  "val": 15
 },
 "IvoryCoast": {
  "buffer_dropped": 11,
  "test": 15,
  "train": 59,
  "val": 15
 }
}
{
 "Amazon": {
  "buffer_dropped": 10,
  "test": 17,
  "test_heldout": 0,
  "train": 54,
  "val": 19
 },
 "Carpathians": {
  "buffer_dropped": 16,
  "test": 17,
  "test_heldout": 0,
  "train": 51,
  "val": 16
 },
 "Congo": {
  "buffer_dropped": 0,
  "test": 0,
  "test_heldout": 100,
  "train": 0,
  "val": 0
 },
 "IvoryCoast": {
  "buffer_dropped": 11,
  "test": 18,
  "test_heldout": 0,
  "train": 54,
  "val": 17
 }
}
```

## lab_classical.txt
```
ndvi_diff t=0.250 {'P': 0.09193012987850076, 'R': 0.2651884906800539, 'F1': 0.13653061468691294, 'IoU': 0.07326689440834254}
sar_ratio t=3.602 {'P': 0.018004282739769284, 'R': 0.169618370287733, 'F1': 0.03255318104974864, 'IoU': 0.01654590138660911}
ndvi_diff t=0.250 {'P': 0.09193012987850076, 'R': 0.2651884906800539, 'F1': 0.13653061468691294, 'IoU': 0.07326689440834254}
sar_ratio t=3.602 {'P': 0.018004282739769284, 'R': 0.169618370287733, 'F1': 0.03255318104974864, 'IoU': 0.01654590138660911}
ndvi_diff t=0.250 {'P': 0.09193012987850076, 'R': 0.2651884906800539, 'F1': 0.13653061468691294, 'IoU': 0.07326689440834254}
sar_ratio t=3.602 {'P': 0.018004282739769284, 'R': 0.169618370287733, 'F1': 0.03255318104974864, 'IoU': 0.01654590138660911}
```

## lab_benchmark.txt
```
full 0.57 M 0.763 GMACs 36.7008357001032 ms
no_cma 0.568 M 0.763 GMACs 36.199060609869775 ms
early_only 0.563 M 0.741 GMACs 44.917446480067156 ms
late_only 0.015 M 0.11 GMACs 12.169935579950106 ms
optical_only 0.569 M 0.741 GMACs 37.953224860357295 ms
sar_only 0.567 M 0.68 GMACs 33.560979000067164 ms
unet_r34 24.483 M 8.227 GMACs 140.1510885398966 ms
unet_plain 1.953 M 10.949 GMACs 183.05086898006266 ms
```

## lab_ivory.txt
```
done IvoryCoast_0_10
done IvoryCoast_0_11
done IvoryCoast_0_14
done IvoryCoast_0_5
done IvoryCoast_11_18
done IvoryCoast_11_19
done IvoryCoast_12_20
done IvoryCoast_13_16
done IvoryCoast_13_2
done IvoryCoast_13_20
done IvoryCoast_14_0
done IvoryCoast_14_11

scenario                                           tiles  NDVI drop in label  outside  contrast
S0 pre 2021 / post 2022 (used in experiments)       12      0.099      0.028      0.071
S1 pre 2022 / post 2023                             12      0.016     -0.003      0.019
S2 pre 2021 / post 2023                             12      0.115      0.026      0.090
```
