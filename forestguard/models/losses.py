"""Loss functions: weighted BCE + Dice (Sørensen–Dice) for class-imbalanced segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1).float()
        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """Binary Focal Loss — down-weights easy negatives."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class CombinedLoss(nn.Module):
    """
    Weighted sum of Binary Cross-Entropy (with pos_weight) and Dice loss.
    Using both BCE and Dice is standard practice for medical/remote-sensing
    segmentation: BCE provides per-pixel supervision while Dice optimises
    the overlap metric directly.
    """

    def __init__(
        self,
        pos_weight: float = 3.0,
        dice_weight: float = 0.5,
        use_focal: bool = False,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = 1.0 - dice_weight

        if use_focal:
            self.bce = FocalLoss()
        else:
            pw = torch.tensor([pos_weight])
            self.bce = lambda logits, targets: F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=pw.to(logits.device)
            )

        self.dice = DiceLoss()

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        bce_loss = self.bce(logits, targets.float())
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
