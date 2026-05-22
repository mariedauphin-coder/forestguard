"""Siamese MobileNetV2 encoder for dual-temporal change detection."""

from __future__ import annotations

import torch
import torch.nn as nn
from timm import create_model
from typing import List


class MobileNetV2Encoder(nn.Module):
    """
    MobileNetV2 feature extractor adapted for arbitrary input channel depth.
    Outputs intermediate feature maps at 4 spatial scales for U-Net skip
    connections (stride 2, 4, 8, 16 relative to input).
    """

    # Feature map channels at each scale for MobileNetV2
    _OUT_CHANNELS = [16, 24, 32, 96, 1280]

    def __init__(
        self,
        in_channels: int = 8,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        backbone = create_model(
            "mobilenetv2_100",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        # Patch the first conv to accept in_channels instead of 3
        first_conv = backbone.conv_stem
        backbone.conv_stem = nn.Conv2d(
            in_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=False,
        )
        if pretrained and in_channels != 3:
            # Initialise extra channels by averaging RGB weights
            with torch.no_grad():
                rgb_weight = first_conv.weight.data  # (C_out, 3, k, k)
                mean_weight = rgb_weight.mean(dim=1, keepdim=True)
                new_weight = mean_weight.repeat(1, in_channels, 1, 1)
                new_weight[:, :3] = rgb_weight
                backbone.conv_stem.weight.data = new_weight

        self.backbone = backbone
        self.out_channels = [24, 32, 96, 320]  # timm feature indices 1-4

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return feature maps at strides [4, 8, 16, 32]."""
        return self.backbone(x)


class SiameseEncoder(nn.Module):
    """
    Shared-weight Siamese encoder: processes 'before' and 'after' images
    through identical MobileNetV2 branches, then concatenates features at
    each scale to encode temporal change.
    """

    def __init__(self, in_channels: int = 8, pretrained: bool = True) -> None:
        super().__init__()
        self.branch = MobileNetV2Encoder(in_channels, pretrained)
        # Output channels are doubled (before + after concatenated)
        self.out_channels = [c * 2 for c in self.branch.out_channels]

    def forward(
        self, before: torch.Tensor, after: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Returns list of difference-encoded feature maps at 4 scales.
        Each element: (B, 2*C, H', W')
        """
        feats_before = self.branch(before)
        feats_after = self.branch(after)
        return [
            torch.cat([fb, fa], dim=1)
            for fb, fa in zip(feats_before, feats_after)
        ]
