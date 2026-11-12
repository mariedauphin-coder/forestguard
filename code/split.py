"""Tile-level, buffered, spatially disjoint split. Whole tiles go to ONE partition; tiles adjacent
(Chebyshev distance <= buffer) to any val/test tile are dropped from training."""
import json, argparse, numpy as np, pandas as pd

def make_split(df, seed=0, val_frac=0.15, test_frac=0.15, buffer=1, holdout_region=None, block=4):
    """Tiles are grouped into block x block spatial blocks; whole blocks are assigned to a partition, so the
    only train/val/test contacts are at block edges, where train tiles within `buffer` tiles are dropped."""
    rng = np.random.RandomState(seed); part = {}
    tiles = df[["region", "tile_id", "tx", "ty"]].drop_duplicates().copy()
    tiles["blk"] = list(zip(tiles.tx // block, tiles.ty // block))
    for reg, g in tiles.groupby("region"):
        if reg == holdout_region:
            for t in g.tile_id: part[(reg, t)] = "test_heldout"
            continue
        blocks = {}; 
        for r in g.itertuples(): blocks.setdefault(r.blk, []).append(r.tile_id)
        keys = list(blocks); rng.shuffle(keys); n = len(g); nt, nv = max(1, round(n * test_frac)), max(1, round(n * val_frac))
        c_t = c_v = 0
        for k in keys:
            m = len(blocks[k])
            if c_t < nt: p = "test"; c_t += m
            elif c_v < nv: p = "val"; c_v += m
            else: p = "train"
            for t in blocks[k]: part[(reg, t)] = p
    xy = {(r.region, r.tile_id): (r.tx, r.ty) for r in tiles.itertuples()}
    held = [k for k, v in part.items() if v in ("val", "test")]
    for k, v in list(part.items()):
        if v != "train": continue
        for h in held:
            if h[0] == k[0] and max(abs(xy[k][0] - xy[h][0]), abs(xy[k][1] - xy[h][1])) <= buffer:
                part[k] = "buffer_dropped"; break
    out = df.copy(); out["part"] = [part[(r, t)] for r, t in zip(df.region, df.tile_id)]
    return out

def summary(df):
    return df.groupby(["region", "part"]).size().unstack(fill_value=0).to_dict("index")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout_region", default=None); ap.add_argument("--buffer", type=int, default=1); ap.add_argument("--block", type=int, default=4)
    a = ap.parse_args(); df = pd.read_csv(f"{a.root}/manifest.csv")
    out = make_split(df, a.seed, buffer=a.buffer, holdout_region=a.holdout_region, block=a.block)
    name = f"split_seed{a.seed}" + (f"_holdout_{a.holdout_region}" if a.holdout_region else "")
    out.to_csv(f"{a.root}/{name}.csv", index=False); print(json.dumps(summary(out), indent=1))
