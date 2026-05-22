"""Segmentation metrics: IoU, F1, precision, recall — computed over batches."""

from __future__ import annotations

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SegmentationMetrics:
    """
    Accumulates TP/FP/FN counts across an epoch then computes macro metrics.
    Call .update() each batch, .compute() at epoch end, .reset() to clear.
    """

    smooth: float = 1e-6
    _tp: float = field(default=0.0, init=False, repr=False)
    _fp: float = field(default=0.0, init=False, repr=False)
    _fn: float = field(default=0.0, init=False, repr=False)
    _total_loss: float = field(default=0.0, init=False, repr=False)
    _n_batches: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._tp = self._fp = self._fn = self._total_loss = 0.0
        self._n_batches = 0

    def update(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        loss: float | None = None,
        threshold: float = 0.5,
    ) -> None:
        preds = (torch.sigmoid(logits.detach()) > threshold).float()
        targets = targets.detach().float()

        self._tp += (preds * targets).sum().item()
        self._fp += (preds * (1 - targets)).sum().item()
        self._fn += ((1 - preds) * targets).sum().item()

        if loss is not None:
            self._total_loss += loss
            self._n_batches += 1

    def compute(self) -> Dict[str, float]:
        tp, fp, fn, s = self._tp, self._fp, self._fn, self.smooth
        precision = (tp + s) / (tp + fp + s)
        recall = (tp + s) / (tp + fn + s)
        f1 = 2 * precision * recall / (precision + recall + s)
        iou = (tp + s) / (tp + fp + fn + s)
        avg_loss = self._total_loss / max(self._n_batches, 1)
        return {
            "iou": iou,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "loss": avg_loss,
        }
