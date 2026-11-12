import ee, numpy as np, pandas as pd
from gee_export import PROJECT, s2, download
ee.Initialize(project=PROJECT)
man = pd.read_csv("raw_tiles/manifest.csv"); man = man[man.region == "IvoryCoast"].sort_values("patch_id")
M = (1, 3); S = 2560; use = []
for r in man.itertuples():
    if int((np.load(f"raw_tiles/{r.file}")["mask"] > 0).sum()) >= 100: use.append(r)
    if len(use) == 12: break
SC = {"S0 pre 2021 / post 2022 (used in experiments)": ("n21", "n22"), "S1 pre 2022 / post 2023": ("n22", "n23"), "S2 pre 2021 / post 2023": ("n21", "n23")}
R = {k: [] for k in SC}
for r in use:
    bb = [r.x0, r.y0, r.x0 + S, r.y0 + S]; g = ee.Geometry.Rectangle(bb, proj=r.crs, geodesic=False)
    img = s2(g, 2021, M).select("NDVI").rename("n21").addBands(s2(g, 2022, M).select("NDVI").rename("n22")).addBands(s2(g, 2023, M).select("NDVI").rename("n23"))
    arr = download(img.toFloat(), bb, crs=r.crs)
    if arr is None or arr["n21"].shape[0] < 256 or arr["n21"].shape[1] < 256: print("skip", r.patch_id); continue
    m = np.load(f"raw_tiles/{r.file}")["mask"] > 0
    A = {n: arr[n][:256, :256].astype(np.float32) for n in ("n21", "n22", "n23")}
    for k, (a, b) in SC.items():
        ok = np.isfinite(A[a]) & np.isfinite(A[b]) & (A[a] != 0) & (A[b] != 0); d = A[a] - A[b]
        if (m & ok).sum() >= 30: R[k].append((d[m & ok].mean(), d[(~m) & ok].mean()))
    print("done", r.patch_id, flush=True)
print(f"\n{'scenario':50} tiles  NDVI drop in label  outside  contrast")
for k, v in R.items():
    v = np.array(v); print(f"{k:50} {len(v):3d}   {v[:,0].mean():8.3f}   {v[:,1].mean():8.3f}   {v[:,0].mean()-v[:,1].mean():8.3f}")
