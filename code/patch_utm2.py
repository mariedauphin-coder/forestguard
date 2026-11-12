p = "tools/gee_export.py"; s = open(p).read()
def sub(old, new, n=1):
    global s
    assert s.count(old) == n, f"NOT FOUND {n}x: {old}"
    s = s.replace(old, new)
UT = '''def utm_tiles(box, crs):
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

def download('''
sub("def download(", UT)
sub("region=ee.Geometry.Rectangle(bbox)", "region=ee.Geometry.Rectangle(bbox, proj=crs, geodesic=False)")
sub('T = tiles(cfg["box"], 0)', 'T = utm_tiles(cfg["box"], cfg["crs"])')
sub("g = ee.Geometry.Rectangle(bb)", 'g = ee.Geometry.Rectangle(bb, proj=cfg["crs"], geodesic=False)')
sub('tx=tx, ty=ty, file=f"{tid}.npz")', 'tx=tx, ty=ty, x0=bb[0], y0=bb[1], crs=cfg["crs"], file=f"{tid}.npz")', n=2)
open(p, "w").write(s); print("patched2")
