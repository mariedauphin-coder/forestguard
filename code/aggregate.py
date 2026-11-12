"""Collect runs -> paper tables (markdown). Mean ± SD over seeds; pooled and macro metrics always labelled."""
import argparse, glob, json, collections, numpy as np, pandas as pd
from metrics import prf, pooled
ap = argparse.ArgumentParser(); ap.add_argument("--runs", default="runs"); ap.add_argument("--bench", default="benchmark.json"); ap.add_argument("--out", default="tables.md"); a = ap.parse_args()
R = [json.load(open(f)) for f in glob.glob(f"{a.runs}/*/result.json")]
bench = json.load(open(a.bench)) if glob.glob(a.bench) else {"models": {}}
ms = lambda v: f"{100*np.mean(v):.1f} ± {100*np.std(v, ddof=1):.1f}" if len(v) > 1 else f"{100*v[0]:.1f}"
by = collections.defaultdict(list)
for r in R: by[(r["variant"], r["split"])].append(r)
out = []
def table(title, variants, split_filter):
    out.append(f"\n### {title}\n\n| Variant | n seeds | P (%) | R (%) | F1 (%) | IoU (%) | macro F1 (%) | macro IoU (%) | Params (M) | MACs (G) | model-only ms | end-to-end ms |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for v in variants:
        for (vv, sp), rs in sorted(by.items()):
            if vv != v or not split_filter(sp): continue
            g = lambda k, m="test_pooled": [r[m][k] for r in rs]
            mf = [r["test_macro"]["empty_correct"]["F1"] for r in rs]; mi = [r["test_macro"]["empty_correct"]["IoU"] for r in rs]
            b = bench["models"].get(v, {})
            f = lambda d, k: f"{d[k]['mean']:.1f} ± {d[k]['sd']:.1f}" if d else "n/a"
            out.append(f"| {v} [{sp}] | {len(rs)} | {ms(g('P'))} | {ms(g('R'))} | {ms(g('F1'))} | {ms(g('IoU'))} | {ms(mf)} | {ms(mi)} | "
                       f"{b.get('params_M', rs[0].get('params',0)/1e6 if rs[0].get('params') else 'n/a')} | {b.get('MACs_G','n/a')} | {f(b.get('model_only_ms'),'mean') if False else (str(round(b['model_only_ms']['mean'],1))+' ± '+str(round(b['model_only_ms']['sd'],1)) if b else 'n/a')} | {(str(round(b['end_to_end_ms']['mean'],1))+' ± '+str(round(b['end_to_end_ms']['sd'],1))) if b else 'n/a'} |")
inreg = lambda s: "holdout" not in s
table("Detection results, in-region test tiles (Table 2)", ["ndvi_diff", "sar_ratio", "unet_plain", "unet_r34", "unet_r34_opt", "full"], inreg)
table("Ablation (Table 4)", ["optical_only", "sar_only", "early_only", "late_only", "no_cma", "full"], inreg)
table("Held-out region transfer (Table 3, last row)", ["full", "unet_r34", "unet_plain"], lambda s: "holdout" in s)
# per-region pooled, in-region, mean over seeds
rows = []
for r_dir in glob.glob(f"{a.runs}/full_*/per_patch_counts.csv"):
    if "holdout" in r_dir: continue
    rows.append(pd.read_csv(r_dir))
if rows:
    out.append("\n### Per-region pooled metrics, ForestGuard full, in-region test tiles (Table 3)\n\n| Region | P (%) | R (%) | F1 (%) | IoU (%) | seeds |\n|---|---|---|---|---|---|")
    regs = sorted(set(pd.concat(rows).region))
    for reg in regs:
        ms_ = [prf(*(int(d[d.region == reg][k].sum()) for k in ("TP", "FP", "FN"))) for d in rows]
        f = lambda k: ms([m[k] for m in ms_])
        out.append(f"| {reg} | {f('P')} | {f('R')} | {f('F1')} | {f('IoU')} | {len(ms_)} |")
open(a.out, "w").write("\n".join(out)); print("\n".join(out))
