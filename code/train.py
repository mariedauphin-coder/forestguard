import argparse, json, os, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from data import PatchSet, compute_stats
from metrics import counts, pooled, macro
from models.forestguard import ForestGuard
from models.baselines import UNetR34, PlainUNet

def build(name, pretrained=False):
    f = dict(modalities="both", use_early=True, use_late=True, use_cma=True)
    V = {"full": {}, "no_cma": dict(use_cma=False), "early_only": dict(use_late=False), "late_only": dict(use_early=False),
         "optical_only": dict(modalities="opt"), "sar_only": dict(modalities="sar")}
    if name in V: return ForestGuard(**{**f, **V[name]}, pretrained=pretrained)
    if name == "unet_r34": return UNetR34("both", pretrained)
    if name == "unet_r34_opt": return UNetR34("opt", pretrained)
    if name == "unet_plain": return PlainUNet("both")
    raise ValueError(name)

def loss_fn(logit, mask, valid, alpha=0.4):
    p = torch.sigmoid(logit)
    bce = (F.binary_cross_entropy_with_logits(logit, mask, reduction="none") * valid).sum() / valid.sum().clamp(min=1)
    pv, tv = p * valid, mask * valid
    dice = 1 - (2 * (pv * tv).sum() + 1) / (pv.sum() + tv.sum() + 1)
    return alpha * bce + (1 - alpha) * dice

@torch.no_grad()
def evaluate(model, loader, dev, save_probs=None):
    model.eval(); rows, ids = [], []
    for sb, ob, sa, oa, m, v, pid in loader:
        pr = torch.sigmoid(model(sb.to(dev), ob.to(dev), sa.to(dev), oa.to(dev))).cpu().numpy()
        for i in range(len(pid)):
            rows.append(counts(pr[i, 0], m[i, 0].numpy(), v[i, 0].numpy())); ids.append(pid[i])
            if save_probs is not None: np.savez_compressed(f"{save_probs}/{pid[i]}.npz", prob=pr[i, 0].astype(np.float16))
    return rows, ids

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--split", required=True, help="split csv name inside root")
    ap.add_argument("--variant", default="full"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40); ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--out", default="runs")
    ap.add_argument("--pretrained", action="store_true"); ap.add_argument("--save_probs", action="store_true")
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--max_patches", type=int, default=0); ap.add_argument("--prior_bias", type=float, default=None)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(f"{a.root}/{a.split}")
    tr, va = df[df.part == "train"], df[df.part == "val"]
    te = df[df.part == "test_heldout"] if (df.part == "test_heldout").any() else df[df.part == "test"]
    if a.max_patches: tr = tr.head(a.max_patches)
    st = compute_stats(a.root, tr)
    run = f"{a.out}/{a.variant}_{a.split.replace('.csv','')}_seed{a.seed}"; os.makedirs(run, exist_ok=True)
    json.dump(st, open(f"{run}/stats.json", "w"))
    L = lambda d, aug, sh: DataLoader(PatchSet(a.root, d, st, aug), batch_size=a.bs, shuffle=sh, num_workers=a.workers)
    tl, vl, el = L(tr, True, True), L(va, False, False), L(te, False, False)
    model = build(a.variant, a.pretrained).to(dev)
    if a.prior_bias is not None: model.head.bias.data.fill_(a.prior_bias)   # start output at the expected positive rate
    npar = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    best, best_ep, t0 = -1, -1, time.time(); hist = []
    for ep in range(a.epochs):
        model.train(); tot = 0
        for sb, ob, sa, oa, m, v, _ in tl:
            sb, ob, sa, oa, m, v = (x.to(dev) for x in (sb, ob, sa, oa, m, v))
            l = loss_fn(model(sb, ob, sa, oa), m, v); opt.zero_grad(); l.backward(); opt.step(); tot += l.item()
        sched.step()
        rows, _ = evaluate(model, vl, dev); f1 = pooled(rows)["F1"]; hist.append(dict(epoch=ep, loss=tot / max(1, len(tl)), val_F1=f1))
        if f1 > best: best, best_ep = f1, ep; torch.save(model.state_dict(), f"{run}/model.pt")
        print(f"ep {ep} loss {hist[-1]['loss']:.4f} valF1 {f1:.4f}", flush=True)
    model.load_state_dict(torch.load(f"{run}/model.pt", map_location=dev))
    sp = None
    if a.save_probs: sp = f"{run}/test_probs"; os.makedirs(sp, exist_ok=True)
    rows, ids = evaluate(model, el, dev, sp)   # test set is evaluated ONCE, with the epoch picked on validation
    pc = pd.DataFrame(rows, columns=["TP", "FP", "FN", "TN"]); pc.insert(0, "patch_id", ids)
    pc = pc.merge(df[["patch_id", "region", "part"]], on="patch_id"); pc.to_csv(f"{run}/per_patch_counts.csv", index=False)
    res = dict(variant=a.variant, seed=a.seed, split=a.split, params=npar, best_epoch=best_ep, val_F1=best,
               test_pooled=pooled(rows), test_macro=macro(rows), epochs=a.epochs, lr=a.lr, bs=a.bs, prior_bias=a.prior_bias,
               n_train=len(tr), n_val=len(va), n_test=len(te), train_seconds=time.time() - t0, device=dev,
               torch=torch.__version__, history=hist)
    json.dump(res, open(f"{run}/result.json", "w"), indent=1); print(json.dumps({k: res[k] for k in ("variant", "seed", "params", "test_pooled")}, indent=1))
