import glob, json, os, sys, numpy as np, pandas as pd
runs = sys.argv[1] if len(sys.argv) > 1 else "runs"
rows = []
for d in sorted(glob.glob(f"{runs}/*/")):
    f = f"{d}per_patch_counts.csv"
    if not os.path.exists(f): continue
    r = json.load(open(f"{d}result.json")); c = pd.read_csv(f)
    for reg, g in c.groupby("region"):
        tp, fp, fn = int(g.TP.sum()), int(g.FP.sum()), int(g.FN.sum())
        rows.append(dict(variant=r["variant"], split=r["split"].replace(".csv", ""), seed=r["seed"], region=reg, tiles=len(g), pos_px=tp + fn,
                         P=100 * tp / max(tp + fp, 1), R=100 * tp / max(tp + fn, 1), F1=100 * 2 * tp / max(2 * tp + fp + fn, 1)))
D = pd.DataFrame(rows)
ms = lambda v: f"{np.mean(v):.1f} ± {np.std(v, ddof=1):.1f}" if len(v) > 1 else f"{v.iloc[0]:.1f}"
for split, S in D.groupby("split"):
    print(f"\n### Per-region pooled metrics, split = {split}\n")
    print("| variant | region | test tiles | positive px | n seeds | P (%) | R (%) | F1 (%) |\n|---|---|---|---|---|---|---|---|")
    for (v, reg), g in S.groupby(["variant", "region"]):
        print(f"| {v} | {reg} | {g.tiles.iloc[0]} | {g.pos_px.iloc[0]} | {len(g)} | {ms(g.P)} | {ms(g.R)} | {ms(g.F1)} |")
