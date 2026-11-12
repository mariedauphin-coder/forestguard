import ee, numpy as np, sys
from gee_export import PROJECT, REGIONS, tiles, s1, s2, label, download
ee.Initialize(project=PROJECT)
name = sys.argv[1] if len(sys.argv) > 1 else "Congo_12_17"
reg, tx, ty = name.split("_"); tx, ty = int(tx), int(ty)
cfg = REGIONS[reg]; Y = cfg["after_year"]; m = cfg["months"]
bb = {(a, b): c for a, b, c in tiles(cfg["box"], 0)}[(tx, ty)]
g = ee.Geometry.Rectangle(bb)
L = label(g, Y)
ee_n = L.select("mask").reduceRegion(ee.Reducer.sum(), g, 10, maxPixels=int(1e9)).getInfo()["mask"]
print(f"{name}: Earth Engine loss px @10m = {int(ee_n)};  saved npz loss px = {int((np.load(f'raw_tiles/{name}.npz')['mask'] > 0).sum())}", flush=True)
def rep(tag, img):
    arr = download(img.toFloat(), bb)
    if arr is None: print(tag, "download failed"); return
    n = arr["mask"]
    print(f"{tag:26} shape={n.shape} bands={len(arr.dtype.names)}  loss px full={int((n > 0.5).sum())}  top-left 256x256={int((n[:256, :256] > 0.5).sum())}  NaN={int(np.isnan(n).sum())}", flush=True)
rep("label only", L)
rep("label + S1 after", s1(g, Y, m).rename(["sa_vv", "sa_vh"]).addBands(L))
rep("label + S2 after", s2(g, Y, m).rename([f"oa_{i}" for i in range(8)]).addBands(L))
full = s1(g, Y - 1, m).rename(["sb_vv", "sb_vh"]).addBands(s2(g, Y - 1, m).rename([f"ob_{i}" for i in range(8)])) \
    .addBands(s1(g, Y, m).rename(["sa_vv", "sa_vh"])).addBands(s2(g, Y, m).rename([f"oa_{i}" for i in range(8)])).addBands(L)
rep("full stack (as exported)", full)
