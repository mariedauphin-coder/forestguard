#!/usr/bin/env python3
"""
Preprocess raw Sentinel-1 and Sentinel-2 scenes and extract training patches.

Usage:
    python scripts/preprocess.py \
        --sar-before   data/raw/sentinel1/before/ \
        --opt-before   data/raw/sentinel2/before/ \
        --sar-after    data/raw/sentinel1/after/ \
        --opt-after    data/raw/sentinel2/after/ \
        --label        data/raw/labels/deforestation.tif \
        --out-dir      data/patches/ \
        --patch-size   256 \
        --stride       128
"""

import argparse
import logging
import numpy as np
from pathlib import Path

from forestguard.preprocessing import FusionPipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ForestGuard preprocessing pipeline")
    parser.add_argument("--sar-before",  required=True)
    parser.add_argument("--opt-before",  required=True)
    parser.add_argument("--sar-after",   required=True)
    parser.add_argument("--opt-after",   required=True)
    parser.add_argument("--label",       required=True)
    parser.add_argument("--out-dir",     required=True)
    parser.add_argument("--patch-size",  type=int, default=256)
    parser.add_argument("--stride",      type=int, default=128)
    args = parser.parse_args()

    out = Path(args.out_dir)
    for sub in ("before", "after", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    pipeline = FusionPipeline(patch_size=args.patch_size, stride=args.stride)
    logger.info("Fusing SAR + optical scenes…")
    before, after, meta = pipeline.fuse_pair(
        args.sar_before, args.opt_before,
        args.sar_after,  args.opt_after,
    )

    import rasterio
    with rasterio.open(args.label) as src:
        label = src.read(1)

    logger.info("Extracting patches…")
    n = 0
    for patch, label_patch, (row, col) in pipeline.extract_patches(before, label):
        stem = f"r{row:05d}_c{col:05d}"
        after_patch = after[:, row : row + args.patch_size, col : col + args.patch_size]
        np.save(out / "before" / f"{stem}.npy", patch)
        np.save(out / "after"  / f"{stem}.npy", after_patch)
        np.save(out / "labels" / f"{stem}.npy", label_patch)
        n += 1

    logger.info("Saved %d patch pairs to %s", n, out)

    from forestguard.training.dataset import ForestChangeDataset
    for split in ("train", "val", "test"):
        ForestChangeDataset.build_manifest(out, out / f"{split}.json", split=split)
        logger.info("Wrote %s manifest", split)


if __name__ == "__main__":
    main()
