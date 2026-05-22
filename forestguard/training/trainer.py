"""Training loop with mixed precision, cosine LR schedule, and TensorBoard logging."""

from __future__ import annotations

import math
import time
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..models.fusion_model import ForestGuardModel
from ..models.losses import CombinedLoss
from .metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: ForestGuardModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: str | Path = "checkpoints",
        epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_epochs: int = 5,
        mixed_precision: bool = True,
        gradient_clip: float = 1.0,
        pos_weight: float = 3.0,
        dice_weight: float = 0.5,
        device: str | torch.device = "cuda",
        log_dir: str | Path = "runs",
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.epochs = epochs
        self.gradient_clip = gradient_clip
        self.mixed_precision = mixed_precision
        self.device = torch.device(device)

        self.criterion = CombinedLoss(pos_weight=pos_weight, dice_weight=dice_weight)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = self._build_scheduler(warmup_epochs, epochs, lr)
        self.scaler = GradScaler(enabled=mixed_precision)
        self.metrics = SegmentationMetrics()
        self.writer = SummaryWriter(log_dir=log_dir)

        self.best_val_iou = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self) -> None:
        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_metrics = self._run_epoch(epoch, train=True)
            val_metrics = self._run_epoch(epoch, train=False)
            elapsed = time.time() - t0

            self._log(epoch, train_metrics, val_metrics, elapsed)

            if val_metrics["iou"] > self.best_val_iou:
                self.best_val_iou = val_metrics["iou"]
                self._save_checkpoint(epoch, val_metrics, best=True)

            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_metrics, best=False)

        self.writer.close()
        logger.info("Training complete. Best val IoU: %.4f", self.best_val_iou)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _run_epoch(self, epoch: int, train: bool) -> dict:
        self.model.train(train)
        self.metrics.reset()
        loader = self.train_loader if train else self.val_loader
        ctx = torch.enable_grad if train else torch.no_grad

        with ctx():
            for before, after, labels in loader:
                before = before.to(self.device, non_blocking=True)
                after = after.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.mixed_precision):
                    logits = self.model(before, after)
                    loss = self.criterion(logits, labels)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                self.metrics.update(logits, labels, loss=loss.item())

        if train:
            self.scheduler.step()

        return self.metrics.compute()

    def _build_scheduler(self, warmup: int, total: int, base_lr: float):
        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return epoch / max(warmup, 1)
            progress = (epoch - warmup) / max(total - warmup, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _log(
        self, epoch: int, train: dict, val: dict, elapsed: float
    ) -> None:
        lr = self.scheduler.get_last_lr()[0]
        logger.info(
            "Epoch %3d/%d | %.1fs | LR %.2e | "
            "Train loss %.4f IoU %.4f | Val loss %.4f IoU %.4f F1 %.4f",
            epoch, self.epochs, elapsed, lr,
            train["loss"], train["iou"],
            val["loss"], val["iou"], val["f1"],
        )
        for key, val_v in val.items():
            self.writer.add_scalar(f"val/{key}", val_v, epoch)
        for key, trn_v in train.items():
            self.writer.add_scalar(f"train/{key}", trn_v, epoch)
        self.writer.add_scalar("lr", lr, epoch)

    def _save_checkpoint(
        self, epoch: int, metrics: dict, best: bool
    ) -> None:
        name = "best.pt" if best else f"epoch_{epoch:03d}.pt"
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "model_config": {
                    "in_channels": 8,
                    "num_classes": 1,
                },
            },
            path,
        )
        logger.info("Saved checkpoint → %s", path)
