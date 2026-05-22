"""ForestGuard: end-to-end Siamese MobileNetV2 + U-Net change detection model."""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Tuple

from .encoder import SiameseEncoder
from .unet import UNetDecoder


class ForestGuardModel(nn.Module):
    """
    Full deforestation detection model.

    Inputs
    ------
    before : (B, C, H, W)   fused SAR+optical at t0
    after  : (B, C, H, W)   fused SAR+optical at t1

    Output
    ------
    logits : (B, 1, H, W)   raw (pre-sigmoid) change map
    """

    def __init__(
        self,
        in_channels: int = 8,
        decoder_channels: Tuple[int, ...] = (256, 128, 64, 32, 16),
        num_classes: int = 1,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = SiameseEncoder(in_channels=in_channels, pretrained=pretrained)
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=list(decoder_channels),
            num_classes=num_classes,
            dropout=dropout,
        )

    def forward(
        self, before: torch.Tensor, after: torch.Tensor
    ) -> torch.Tensor:
        input_size = before.shape[2:]
        features = self.encoder(before, after)
        return self.decoder(features, input_size)

    def predict_mask(
        self,
        before: torch.Tensor,
        after: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Return binary (0/1) mask after sigmoid thresholding."""
        with torch.no_grad():
            logits = self(before, after)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, **kwargs) -> "ForestGuardModel":
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model_cfg = ckpt.get("model_config", {})
        model_cfg.update(kwargs)
        model = cls(**model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        return model
