p = "tools/gee_export.py"; s = open(p).read()
def sub(old, new):
    global s
    assert s.count(old) == 1, f"NOT FOUND exactly once: {old}"
    s = s.replace(old, new)
sub("def download(img, bbox, tries=4):", 'def download(img, bbox, tries=4, crs="EPSG:4326"):')
sub('scale=SCALE, crs="EPSG:4326", format="NPY"', 'scale=SCALE, crs=crs, format="NPY"')
sub("arr = download(stack, bb)", 'arr = download(stack, bb, crs=cfg["crs"])')
sub("box=[-63.2, -9.8, -62.6, -9.3],", 'box=[-63.2, -9.8, -62.6, -9.3], crs="EPSG:32720",')
sub("box=[23.5, -1.0, 24.1, -0.5],", 'box=[23.5, -1.0, 24.1, -0.5], crs="EPSG:32734",')
sub("box=[-6.6, 5.9, -6.0, 6.4],", 'box=[-6.6, 5.9, -6.0, 6.4], crs="EPSG:32629",')
sub("box=[24.8, 45.6, 25.4, 46.0],", 'box=[25.3, 47.3, 25.9, 47.7], crs="EPSG:32635",')
open(p, "w").write(s); print("patched")
