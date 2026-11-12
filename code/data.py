"""Patch folder format
root/manifest.csv : patch_id,region,tile_id,tx,ty,file
root/<file>.npz   : sar_b (2,H,W) dB, opt_b (8,H,W) [B2,B3,B4,B5,B8,B11,NDVI,NDMI], sar_a, opt_a, mask (H,W) 0/1, valid (H,W) 0/1
tx,ty = integer grid indices of the tile inside its region (used for the buffered split)."""
import json, numpy as np, pandas as pd
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:   # lets split/classical/synthetic run without torch
    torch = None; Dataset = object

KEYS = ["sar_b", "opt_b", "sar_a", "opt_a"]

def load_npz(path):
    z = np.load(path); out = {k: z[k].astype(np.float32) for k in KEYS}
    out["mask"] = z["mask"].astype(np.float32)
    out["valid"] = (z["valid"] if "valid" in z.files else np.ones_like(z["mask"])).astype(np.float32)
    return out

def compute_stats(root, df):
    """Per-channel mean/std from TRAIN patches only; pre and post share statistics per modality."""
    s = {"sar": [0, 0, 0], "opt": [0, 0, 0]}
    for f in df.file:
        d = load_npz(f"{root}/{f}")
        for m in ("sar", "opt"):
            x = np.concatenate([d[m + "_b"], d[m + "_a"]], 2)       # C,H,2W
            c = x.shape[0]
            if isinstance(s[m][0], int): s[m] = [np.zeros(c), np.zeros(c), 0]
            s[m][0] += x.reshape(c, -1).sum(1); s[m][1] += (x.reshape(c, -1) ** 2).sum(1); s[m][2] += x.shape[1] * x.shape[2]
    st = {}
    for m in s:
        n = s[m][2]; mean = s[m][0] / n; var = s[m][1] / n - mean ** 2
        st[m] = {"mean": mean.tolist(), "std": np.sqrt(np.maximum(var, 1e-8)).tolist()}
    return st

class PatchSet(Dataset):
    def __init__(self, root, df, stats, augment=False, raw=False):
        self.root, self.df, self.st, self.aug, self.raw = root, df.reset_index(drop=True), stats, augment, raw
    def __len__(self): return len(self.df)
    def _norm(self, x, m):
        mean = np.array(self.st[m]["mean"], np.float32)[:, None, None]; std = np.array(self.st[m]["std"], np.float32)[:, None, None]
        return (x - mean) / std
    def __getitem__(self, i):
        r = self.df.iloc[i]; d = load_npz(f"{self.root}/{r.file}")
        if not self.raw:
            for k in KEYS: d[k] = self._norm(d[k], k[:3])
        if self.aug:
            k = np.random.randint(4); fl = np.random.rand() < 0.5
            for key in list(d): 
                a = d[key]; a = np.rot90(a, k, axes=(-2, -1)); a = a[..., ::-1] if fl else a; d[key] = np.ascontiguousarray(a)
        t = lambda k: torch.from_numpy(d[k])
        return t("sar_b"), t("opt_b"), t("sar_a"), t("opt_a"), t("mask")[None], t("valid")[None], str(r.patch_id)
