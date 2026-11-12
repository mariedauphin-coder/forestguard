import os
"""Earth Engine export of before/after Sentinel-1 + Sentinel-2 tiles with Hansen GFC loss labels.
NOT TESTED against live Earth Engine from the authoring sandbox (no network). Run it, inspect 2-3 tiles
visually, then scale up.  Setup:  pip install earthengine-api requests ; earthengine authenticate ; set PROJECT.

Labels: Hansen Global Forest Change (UMD/hansen/global_forest_change_2025_v1_13), loss in the AFTER year,
tree cover 2000 > 30 %. Pre/post images are median composites of the same calendar window in year Y-1 and Y.
Each tile = TILE_PX x TILE_PX pixels at 10 m, non-overlapping on a grid (tx,ty = grid index)."""
import ee, io, os, json, math, time, argparse, requests, numpy as np, zipfile

PROJECT = os.environ.get("EE_PROJECT", "YOUR-EE-PROJECT")
# lon/lat boxes: [west, south, east, north]. EDIT: choose forested areas; keep boxes ~30-60 km wide.
REGIONS = {
    "Amazon":      dict(box=[-63.2, -9.8, -62.6, -9.3], crs="EPSG:32720",  after_year=2022, months=(6, 9)),
    "Congo":       dict(box=[23.5, -1.0, 24.1, -0.5], crs="EPSG:32734",    after_year=2022, months=(6, 8)),
    "IvoryCoast":  dict(box=[-6.6, 5.9, -6.0, 6.4], crs="EPSG:32629",      after_year=2022, months=(1, 3)),   # window may cross year end: handled below
    "Carpathians": dict(box=[25.3, 47.3, 25.9, 47.7], crs="EPSG:32635",    after_year=2022, months=(6, 9)),
}
TILE_PX, SCALE = 256, 10

def window(year, m):
    a, b = m
    start = ee.Date.fromYMD(year, a, 1)
    end = ee.Date.fromYMD(year if b >= a else year + 1, b, 28).advance(1, "day")
    return start, end

def s1(geom, year, m):
    s, e = window(year, m)
    c = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filterDate(s, e).filter(ee.Filter.eq("instrumentMode", "IW"))
         .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))
    return c.select(["VV", "VH"]).median()            # GRD in GEE is already dB

def s2(geom, year, m):
    s, e = window(year, m)
    def mask(i):
        scl = i.select("SCL"); bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11))
        return i.updateMask(bad.Not())
    c = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(geom).filterDate(s, e).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40)).map(mask)
    img = c.select(["B2", "B3", "B4", "B5", "B8", "B11"]).median().divide(10000)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI"); ndmi = img.normalizedDifference(["B8", "B11"]).rename("NDMI")
    return img.addBands([ndvi, ndmi]).unmask(0)    # persistent-cloud gaps become 0 (documented limitation)

def label(geom, year):
    h = ee.Image("UMD/hansen/global_forest_change_2025_v1_13")
    loss = h.select("lossyear").eq(year - 2000).And(h.select("treecover2000").gt(30)).rename("mask")
    valid = h.select("datamask").eq(1).rename("valid")
    return loss.addBands(valid)

def tiles(box, lat0):
    w, s, e, n = box; dlat = TILE_PX * SCALE / 111320.0
    out = []; ty = 0; lat = s
    while lat + dlat <= n:
        dlon = TILE_PX * SCALE / (111320.0 * math.cos(math.radians(lat + dlat / 2))); tx = 0; lon = w
        while lon + dlon <= e:
            out.append((tx, ty, [lon, lat, lon + dlon, lat + dlat])); lon += dlon; tx += 1
        lat += dlat; ty += 1
    return out

def utm_tiles(box, crs):
    """Non-overlapping TILE_PX x TILE_PX tiles on a 10 m grid in the region's UTM zone. Tile = square [x0,y0,x0+S,y0+S] in metres."""
    from pyproj import Transformer
    w, s_, e, n = box; tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    pts = [tr.transform(lo, la) for lo in (w, e) for la in (s_, n)]
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]; S = TILE_PX * SCALE
    X0 = math.ceil(min(xs) / S) * S; Y0 = math.ceil(min(ys) / S) * S; out = []; ty = 0; y = Y0
    while y + S <= max(ys):
        tx = 0; x = X0
        while x + S <= max(xs):
            out.append((tx, ty, [x, y, x + S, y + S])); x += S; tx += 1
        y += S; ty += 1
    return out

def download(img, bbox, tries=4, crs="EPSG:4326"):
    for k in range(tries):
        try:
            url = img.getDownloadURL(dict(region=ee.Geometry.Rectangle(bbox, proj=crs, geodesic=False), scale=SCALE, crs=crs, format="NPY"))
            r = requests.get(url, timeout=180); r.raise_for_status(); return np.load(io.BytesIO(r.content), allow_pickle=True)
        except Exception as ex:
            print("retry", k, ex); time.sleep(3 * (k + 1))
    return None

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="raw_tiles"); ap.add_argument("--per_region", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    ee.Initialize(project=PROJECT); os.makedirs(a.out, exist_ok=True); rng = np.random.RandomState(a.seed); manifest = []
    for reg, cfg in REGIONS.items():
        T = utm_tiles(cfg["box"], cfg["crs"]); rng.shuffle(T); T = T[: a.per_region]; Y = cfg["after_year"]
        for tx, ty, bb in T:
            tid = f"{reg}_{tx}_{ty}"; fn = f"{a.out}/{tid}.npz"
            if os.path.exists(fn): manifest.append(dict(patch_id=tid, region=reg, tile_id=tid, tx=tx, ty=ty, x0=bb[0], y0=bb[1], crs=cfg["crs"], file=f"{tid}.npz")); continue
            g = ee.Geometry.Rectangle(bb, proj=cfg["crs"], geodesic=False)
            stack = s1(g, Y - 1, cfg["months"]).rename(["sb_vv", "sb_vh"]).addBands(s2(g, Y - 1, cfg["months"]).rename([f"ob_{i}" for i in range(8)])) \
                .addBands(s1(g, Y, cfg["months"]).rename(["sa_vv", "sa_vh"])).addBands(s2(g, Y, cfg["months"]).rename([f"oa_{i}" for i in range(8)])).addBands(label(g, Y)).toFloat()
            arr = download(stack, bb, crs=cfg["crs"])
            if arr is None: print("skip", tid); continue
            names = arr.dtype.names; A = np.stack([arr[n] for n in names]).astype(np.float32); idx = {n: i for i, n in enumerate(names)}
            bad = ~np.isfinite(A); A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
            okpix = ~bad[[idx[n] for n in ("sb_vv", "sb_vh", "sa_vv", "sa_vh")]].any(0)
            H = W = TILE_PX; crop = lambda L: np.stack([A[idx[n]][:H, :W] for n in L])
            if A.shape[1] < H or A.shape[2] < W: print("too small", tid, A.shape); continue
            np.savez_compressed(fn, sar_b=crop(["sb_vv", "sb_vh"]), opt_b=crop([f"ob_{i}" for i in range(8)]), sar_a=crop(["sa_vv", "sa_vh"]),
                                opt_a=crop([f"oa_{i}" for i in range(8)]), mask=(crop(["mask"])[0] > 0.5).astype(np.uint8), valid=((crop(["valid"])[0] > 0.5) & okpix[:H, :W]).astype(np.uint8))
            manifest.append(dict(patch_id=tid, region=reg, tile_id=tid, tx=tx, ty=ty, x0=bb[0], y0=bb[1], crs=cfg["crs"], file=f"{tid}.npz")); print("ok", tid, "loss px", int(crop(["mask"]).sum()))
    import pandas as pd; pd.DataFrame(manifest).to_csv(f"{a.out}/manifest.csv", index=False); print(len(manifest), "tiles")
