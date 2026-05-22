#!/usr/bin/env python3
"""
Train ForestGuard model.

Usage:
    python scripts/train.py \
        --train-manifest  data/patches/train.json \
        --val-manifest    data/patches/val.json \
        --checkpoint-dir  checkpoints/ \
        --epochs 100 \
        --batch-size 16 \
        --lr 1e-4 \
        --device cuda
"""

import argparse
import logging
import torch
from torch.utils.data import DataLoader

from forestguard.models import ForestGuardModel
from forestguard.training import ForestChangeDataset, Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="ForestGuard training")
    parser.add_argument("--train-manifest",  required=True)
    parser.add_argument("--val-manifest",    required=True)
    parser.add_argument("--checkpoint-dir",  default="checkpoints/")
    parser.add_argument("--log-dir",         default="runs/")
    parser.add_argument("--epochs",          type=int,   default=100)
    parser.add_argument("--batch-size",      type=int,   default=16)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--weight-decay",    type=float, default=1e-5)
    parser.add_argument("--pos-weight",      type=float, default=3.0)
    parser.add_argument("--dice-weight",     type=float, default=0.5)
    parser.add_argument("--warmup-epochs",   type=int,   default=5)
    parser.add_argument("--no-amp",          action="store_true")
    parser.add_argument("--workers",         type=int,   default=4)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--pretrained",      action="store_true", default=True)
    parser.add_argument("--no-pretrained",   dest="pretrained", action="store_false")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    train_ds = ForestChangeDataset(args.train_manifest, augment=True)
    val_ds = ForestChangeDataset(args.val_manifest, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )

    model = ForestGuardModel(pretrained=args.pretrained)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        mixed_precision=not args.no_amp,
        pos_weight=args.pos_weight,
        dice_weight=args.dice_weight,
        device=device,
    )

    trainer.fit()


if __name__ == "__main__":
    main()
