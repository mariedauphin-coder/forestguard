"""Non-neural reference baselines on the SAME test tiles. Thresholds are tuned on validation tiles only.
  ndvi_diff : optical-only, change = NDVI(before) - NDVI(after) > t
  sar_ratio : SAR-only, change = |VH_after - VH_before| (dB) > t
These are NOT GLAD-L or CuSum; the paper must call them what they are."""
import argparse, json, os, numpy as np, pandas as pd
from data import PatchSet
from metrics import counts, pooled, macro

def score(kind, d):
    if kind == "ndvi_diff": return d["opt_b"][6] - d["opt_a"][6]
    return np.abs(d["sar_a"][1] - d["sar_b"][1])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--out", default="runs"); a = ap.parse_args()
    from data import load_npz
    df = pd.read_csv(f"{a.root}/{a.split}"); va = df[df.part == "val"]; te = df[df.part == "test_heldout"] if (df.part == "test_heldout").any() else df[df.part == "test"]
    for kind in ("ndvi_diff", "sar_ratio"):
        V = [load_npz(f"{a.root}/{f}") for f in va.file]; S = [score(kind, d) for d in V]
        cand = np.quantile(np.concatenate([s.ravel() for s in S]), np.linspace(0.5, 0.999, 60))
        best = max(cand, key=lambda t: pooled([counts((s >= t).astype(float), d["mask"], d["valid"]) for s, d in zip(S, V)])["F1"])
        rows, ids = [], []
        for f, pid in zip(te.file, te.patch_id):
            d = load_npz(f"{a.root}/{f}"); rows.append(counts((score(kind, d) >= best).astype(float), d["mask"], d["valid"])); ids.append(pid)
        run = f"{a.out}/{kind}_{a.split.replace('.csv','')}_seed0"; os.makedirs(run, exist_ok=True)
        pc = pd.DataFrame(rows, columns=["TP", "FP", "FN", "TN"]); pc.insert(0, "patch_id", ids)
        pc.merge(df[["patch_id", "region", "part"]], on="patch_id").to_csv(f"{run}/per_patch_counts.csv", index=False)
        json.dump(dict(variant=kind, seed=0, split=a.split, params=0, threshold=float(best), test_pooled=pooled(rows), test_macro=macro(rows)), open(f"{run}/result.json", "w"), indent=1)
        print(kind, "t=%.3f" % best, pooled(rows))
