import argparse, glob, json, os, numpy as np, pandas as pd
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--exclude", default="IvoryCoast"); ap.add_argument("--K", type=int, nargs="+", default=[100, 500]); a = ap.parse_args()
L = []
for d in sorted(glob.glob(f"{a.runs}/*/")):
    f = f"{d}per_patch_counts.csv"
    if not os.path.exists(f): continue
    r = json.load(open(f"{d}result.json")); c = pd.read_csv(f); c["variant"] = r["variant"]; c["split"] = r["split"].replace(".csv", ""); c["seed"] = r["seed"]; L.append(c)
C = pd.concat(L); ms = lambda v: f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}" if len(v) > 1 else f"{v.iloc[0]:.1f}"
def pix(g):
    tp, fp, fn = g.TP.sum(), g.FP.sum(), g.FN.sum()
    return dict(P=100 * tp / max(tp + fp, 1), R=100 * tp / max(tp + fn, 1), F1=100 * 2 * tp / max(2 * tp + fp + fn, 1), IoU=100 * tp / max(tp + fp + fn, 1))
def alert(K):
    def f(g):
        t = (g.TP + g.FN) >= K; p = (g.TP + g.FP) >= K; tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
        return dict(P=100 * tp / max(tp + fp, 1), R=100 * tp / max(tp + fn, 1), FDP=100 * fp / max(tp + fp, 1), F1=100 * 2 * tp / max(2 * tp + fp + fn, 1), true_alert_tiles=tp + fn)
    return f
def show(title, data, fn):
    print(f"\n### {title}\n"); rows = []
    for (sp, v), g in data.groupby(["split", "variant"]):
        per = pd.DataFrame([fn(h) for _, h in g.groupby("seed")]); cols = list(per.columns)
        if not rows: print("| split | variant | n seeds | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 3)); rows.append(1)
        print(f"| {sp} | {v} | {len(per)} | " + " | ".join(ms(per[k]) for k in cols) + " |")
X = C[C.region != a.exclude]
show("Pooled pixel metrics, ALL regions (should equal aggregate.py)", C, pix)
show(f"Pooled pixel metrics EXCLUDING {a.exclude} (post-hoc, no retraining)", X, pix)
for K in a.K:
    show(f"Tile alerts, K = {K} px (alert if predicted/true loss px in the tile >= K), ALL regions", C, alert(K))
    show(f"Tile alerts, K = {K} px, EXCLUDING {a.exclude} (post-hoc)", X, alert(K))
