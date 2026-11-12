import ee, glob, numpy as np, collections
from gee_export import PROJECT, REGIONS, tiles
ee.Initialize(project=PROJECT)
h = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
forest = h.select("treecover2000").gt(30)
loss = h.select("lossyear").eq(22).And(forest).rename("loss")
G = {k: {(tx, ty): bb for tx, ty, bb in tiles(v["box"], 0)} for k, v in REGIONS.items()}
R = collections.defaultdict(lambda: [0.0, 0.0, 0.0, 0])
for f in sorted(glob.glob("raw_tiles/*.npz")):
    name = f.split("/")[-1][:-4]; reg, tx, ty = name.split("_")[0], *map(int, name.split("_")[1:])
    if R[reg][3] >= 10: continue
    g = ee.Geometry.Rectangle(G[reg][(tx, ty)])
    e10 = loss.reduceRegion(ee.Reducer.mean(), g, 10, maxPixels=int(1e9)).getInfo()["loss"] or 0
    e30 = loss.reduceRegion(ee.Reducer.mean(), g, 30, maxPixels=int(1e9)).getInfo()["loss"] or 0
    sv = float((np.load(f)["mask"] > 0).mean())
    R[reg][0] += e10; R[reg][1] += e30; R[reg][2] += sv; R[reg][3] += 1
    print(f"{name:22} EE@10m={e10*100:6.2f}%  EE@30m={e30*100:6.2f}%  saved={sv*100:6.2f}%", flush=True)
print("\nmean over the first 10 tiles per region:")
for r, (a, b, c, n) in R.items():
    print(f"{r:12} EE@10m={a/n*100:6.2f}%  EE@30m={b/n*100:6.2f}%  saved={c/n*100:6.2f}%")
print("\nshare of forest pixels lost, by year (whole box, 30 m):")
for r, v in REGIONS.items():
    hist = h.select("lossyear").updateMask(forest).reduceRegion(ee.Reducer.frequencyHistogram(), ee.Geometry.Rectangle(v["box"]), 30, maxPixels=int(1e9)).getInfo()["lossyear"]
    tot = sum(hist.values())
    print(f"{r:12}", "  ".join(f"20{y}:{hist.get(str(y), 0)/tot*100:4.2f}%" for y in range(18, 25)))
