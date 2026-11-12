"""SYNTHETIC data to test the code. Results from this data are meaningless and must never appear in the paper."""
import argparse, os, numpy as np, pandas as pd
from scipy.ndimage import gaussian_filter
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="synth"); ap.add_argument("--tiles", type=int, default=24); ap.add_argument("--size", type=int, default=64); a = ap.parse_args()
rng = np.random.RandomState(0); os.makedirs(a.out, exist_ok=True); rows = []; S = a.size
for reg in ["Amazon", "Congo", "IvoryCoast", "Carpathians"]:
    side = int(np.ceil(np.sqrt(a.tiles)))
    for k in range(a.tiles):
        tx, ty = k % side, k // side; tid = f"{reg}_{tx}_{ty}"
        base = gaussian_filter(rng.randn(S, S), 3)
        m = np.zeros((S, S), np.uint8)
        for _ in range(rng.randint(0, 3)):
            x0, y0, w, h = rng.randint(0, S - 16), rng.randint(0, S - 16), rng.randint(6, 16), rng.randint(6, 16); m[y0:y0 + h, x0:x0 + w] = 1
        ndvi_b = 0.8 + 0.05 * base; ndvi_a = ndvi_b - 0.45 * m + 0.03 * rng.randn(S, S)
        mk = lambda c, f: np.stack([f + 0.02 * rng.randn(S, S) for _ in range(c)])
        ob = np.concatenate([mk(6, 0.1 + 0.02 * base), ndvi_b[None], (0.4 + 0 * base)[None]]).astype(np.float32)
        oa = np.concatenate([mk(6, 0.1 + 0.02 * base + 0.08 * m), ndvi_a[None], (0.4 - 0.2 * m)[None]]).astype(np.float32)
        if rng.rand() < 0.4: oa = oa + rng.randn(*oa.shape).astype(np.float32) * 0.2   # "cloudy" after image: optical useless
        sb = (-12 + base[None] * 0.5 + 0.8 * rng.randn(2, S, S)).astype(np.float32); sa = (sb - 3.0 * m[None] + 0.8 * rng.randn(2, S, S)).astype(np.float32)
        f = f"{tid}.npz"; np.savez_compressed(f"{a.out}/{f}", sar_b=sb, opt_b=ob, sar_a=sa, opt_a=oa, mask=m, valid=np.ones_like(m))
        rows.append(dict(patch_id=tid, region=reg, tile_id=tid, tx=tx, ty=ty, file=f))
pd.DataFrame(rows).to_csv(f"{a.out}/manifest.csv", index=False); print(len(rows), "synthetic tiles")
