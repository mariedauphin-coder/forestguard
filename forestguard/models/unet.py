"""U-Net decoder with skip connections from the Siamese encoder."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=kernel // 2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample → concatenate skip → double conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """
    Progressive upsampling decoder that consumes Siamese encoder skips
    (ordered finest-to-coarsest) and produces a full-resolution logit map.
    """

    def __init__(
        self,
        encoder_channels: List[int],
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        num_classes: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        # encoder_channels: coarsest to finest (reversed skips)
        enc_ch = list(reversed(encoder_channels))

        blocks = []
        in_ch = enc_ch[0]
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = enc_ch[i + 1] if i + 1 < len(enc_ch) else 0
            blocks.append(DecoderBlock(in_ch, skip_ch, out_ch))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        self.dropout = nn.Dropout2d(dropout)
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

    def forward(
        self, features: List[torch.Tensor], input_size: tuple
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features   : list from SiameseEncoder, index 0 = finest, -1 = coarsest
        input_size : (H, W) of the original network input
        """
        skips = list(reversed(features))   # coarsest first
        x = skips[0]
        for i, block in enumerate(self.blocks):
            skip = skips[i + 1] if i + 1 < len(skips) else None
            x = block(x, skip)

        x = self.dropout(x)
        x = self.head(x)

        if x.shape[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x
