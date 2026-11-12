"""Controlled CPU benchmark: same hardware, threads, input, batch, precision, warm-up for every model.
Reports model-only and end-to-end (normalise + forward + sigmoid + alert pipeline) latency, params, MACs."""
import argparse, json, platform, time, numpy as np, torch
from train import build

def cpu_model():
    try: return [l.split(":")[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
    except Exception: return platform.processor()

def macs(model, x):
    from torch.utils.flop_counter import FlopCounterMode
    with FlopCounterMode(display=False) as fc: model(*x)
    return fc.get_total_flops() / 2

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--variants", nargs="+", default=["full", "no_cma", "early_only", "late_only", "optical_only", "sar_only", "unet_r34", "unet_plain"])
    ap.add_argument("--threads", type=int, default=4); ap.add_argument("--warmup", type=int, default=20); ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--size", type=int, default=256); ap.add_argument("--out", default="benchmark.json"); a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(0)
    import alert_pipeline as ap_; from affine import Affine
    tf = Affine(0.0001, 0, -60.0, 0, -0.0001, -2.5); S = a.size
    raw = [np.random.randn(2, S, S).astype(np.float32), np.random.randn(8, S, S).astype(np.float32)] * 2
    res = dict(cpu=cpu_model(), threads=a.threads, torch=torch.__version__, input=[1, S, S], precision="fp32", batch=1, warmup=a.warmup, reps=a.reps, models={})
    for v in a.variants:
        m = build(v).eval(); x = [torch.from_numpy(r)[None] for r in raw]
        npar = sum(p.numel() for p in m.parameters()); mac = macs(m, x)
        mo, e2e = [], []
        with torch.no_grad():
            for i in range(a.warmup + a.reps):
                t0 = time.perf_counter(); m(*x); t1 = time.perf_counter()
                xs = [torch.from_numpy((r - r.mean()) / (r.std() + 1e-6))[None] for r in raw]
                p = torch.sigmoid(m(*xs))[0, 0].numpy(); ap_.generate_alerts(p, tf); t2 = time.perf_counter()
                if i >= a.warmup: mo.append((t1 - t0) * 1000); e2e.append((t2 - t1 + (t1 - t0)) * 1000 if False else (t2 - t1) * 1000)
        st = lambda L: dict(mean=float(np.mean(L)), sd=float(np.std(L, ddof=1)), p50=float(np.percentile(L, 50)), p95=float(np.percentile(L, 95)))
        res["models"][v] = dict(params=npar, params_M=round(npar / 1e6, 3), MACs_G=round(mac / 1e9, 3), model_only_ms=st(mo), end_to_end_ms=st(e2e))
        print(v, res["models"][v]["params_M"], "M", res["models"][v]["MACs_G"], "GMACs", res["models"][v]["model_only_ms"]["mean"], "ms", flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
