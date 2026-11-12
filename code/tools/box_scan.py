import ee
from gee_export import PROJECT, REGIONS, tiles
ee.Initialize(project=PROJECT)
h = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
forest = h.select("treecover2000").unmask(0).gt(30).rename("forest")
loss = h.select("lossyear").unmask(0).eq(22).And(forest).rename("loss")
bb = {(a, b): c for a, b, c in tiles(REGIONS["Congo"]["box"], 0)}[(12, 17)]
v = loss.reduceRegion(ee.Reducer.mean(), ee.Geometry.Rectangle(bb), 10, maxPixels=int(1e9)).getInfo()["loss"]
print(f"check Congo_12_17: loss share of area = {v*100:.2f}%  (expect about 0.55%)\n", flush=True)
C = {f"{k} (current)": x["box"] for k, x in REGIONS.items()}
C.update({
    "Bucovina":     [25.3, 47.3, 25.9, 47.7],
    "Bistrita":     [25.8, 46.8, 26.4, 47.2],
    "Harghita":     [25.5, 46.4, 26.1, 46.8],
    "Vrancea":      [26.0, 45.8, 26.6, 46.2],
    "Maramures":    [24.0, 47.6, 24.6, 48.0],
    "Arges-Valcea": [24.4, 45.2, 25.0, 45.6],
    "Zakarpattia":  [23.2, 48.2, 23.8, 48.6],
})
for n, b in C.items():
    r = loss.addBands(forest).reduceRegion(ee.Reducer.mean(), ee.Geometry.Rectangle(b), 30, maxPixels=int(1e9)).getInfo()
    f, l = r["forest"] or 0, r["loss"] or 0
    print(f"{n:28} forest={f*100:5.1f}% of area   loss2022={l*100:5.2f}% of area   ({l/max(f,1e-9)*100:4.2f}% of forest)", flush=True)
from gee_export import label, download
bb = {(a, b): c for a, b, c in tiles(REGIONS["Carpathians"]["box"], 0)}[(0, 1)]
arr = download(label(ee.Geometry.Rectangle(bb), 2022).toFloat(), bb)
print("\nCarpathians tile array shape (rows, cols):", arr["mask"].shape)
