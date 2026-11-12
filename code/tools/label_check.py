import numpy as np, glob, collections
R = collections.defaultdict(list)
for f in sorted(glob.glob("raw_tiles/*.npz")):
    z = np.load(f); m = z["mask"] > 0; v = z["valid"] > 0
    ok = v & (z["opt_a"][6] != 0) & (z["opt_b"][6] != 0)
    if (m & ok).sum() < 30: continue
    dn = z["opt_b"][6] - z["opt_a"][6]; dv = z["sar_a"][1] - z["sar_b"][1]
    R[f.split("/")[-1].split("_")[0]].append((dn[m & ok].mean(), dn[(~m) & ok].mean(), dv[m & v].mean(), dv[(~m) & v].mean()))
print(f"{'region':14}{'tiles':>6}{'NDVIdrop_in':>13}{'NDVIdrop_out':>14}{'VHchg_in':>10}{'VHchg_out':>11}")
for k, L in R.items():
    a = np.mean(L, 0); print(f"{k:14}{len(L):6d}{a[0]:13.3f}{a[1]:14.3f}{a[2]:10.2f}{a[3]:11.2f}")
