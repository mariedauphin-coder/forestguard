import numpy as np
def counts(prob, mask, valid, thr=0.5):
    """All arrays (H,W). Ignored pixels (valid==0) are excluded from every count."""
    p = (prob >= thr) & (valid > 0); t = (mask > 0.5) & (valid > 0); v = valid > 0
    tp = int((p & t).sum()); fp = int((p & ~t).sum()); fn = int((~p & t & v).sum()); tn = int((~p & ~t & v).sum())
    return tp, fp, fn, tn
def prf(tp, fp, fn):
    P = tp / (tp + fp) if tp + fp else 0.0; R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0; I = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    return dict(P=P, R=R, F1=F, IoU=I)
def pooled(rows):
    tp, fp, fn = (sum(r[i] for r in rows) for i in (0, 1, 2)); return prf(tp, fp, fn)
def macro(rows):
    """Per-patch metrics averaged with equal weight. A patch with empty reference AND empty prediction
    scores 1 on all metrics ('empty_correct'); 'nonempty' excludes such patches. Returns both."""
    allv, nonv = [], []
    for tp, fp, fn, tn in rows:
        if tp + fp + fn == 0: allv.append(dict(P=1, R=1, F1=1, IoU=1)); continue
        m = prf(tp, fp, fn); allv.append(m); nonv.append(m)
    avg = lambda L: {k: float(np.mean([x[k] for x in L])) if L else float("nan") for k in ("P", "R", "F1", "IoU")}
    return {"empty_correct": avg(allv), "nonempty": avg(nonv), "n_patches": len(rows), "n_empty": len(rows) - len(nonv)}
