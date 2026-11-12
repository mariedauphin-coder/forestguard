import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import mobilenet_v2
from .fusion import LateBranch, DSConv

class SiameseEncoder(nn.Module):
    def __init__(self, in_ch, pretrained=False):
        super().__init__()
        f = mobilenet_v2(weights="DEFAULT" if pretrained else None).features
        old = f[0][0]; new = nn.Conv2d(in_ch, old.out_channels, 3, 2, 1, bias=False)
        if pretrained:
            with torch.no_grad(): new.weight.copy_(old.weight.mean(1, keepdim=True).repeat(1, in_ch, 1, 1))
        f[0][0] = new
        self.s1, self.s2, self.s3, self.s4 = f[0:2], f[2:4], f[4:7], f[7:14]
        self.chs = [16, 24, 32, 96]
    def forward(self, x):
        a = self.s1(x); b = self.s2(a); c = self.s3(b); d = self.s4(c); return [a, b, c, d]

class Dec(nn.Module):
    def __init__(self, i, s, o): super().__init__(); self.c = DSConv(i + s, o)
    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]: x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], 1)
        return self.c(x)

class ForestGuard(nn.Module):
    """Variants: use_early / use_late / use_cma, and modalities ('both','opt','sar').
    Inputs: sar_b, opt_b, sar_a, opt_a  (B,2/8,H,W). Unused modality tensors are ignored."""
    def __init__(self, modalities="both", use_early=True, use_late=True, use_cma=True, pretrained=False):
        super().__init__()
        assert use_early or use_late
        self.mod, self.use_early, self.use_late = modalities, use_early, use_late
        sc = 2 if modalities in ("both", "sar") else 0; oc = 8 if modalities in ("both", "opt") else 0
        if use_early:
            self.enc = SiameseEncoder(sc + oc, pretrained); ch = self.enc.chs
        if use_late:
            self.late = LateBranch(sc, oc, 96, use_cma)
        if use_early:
            self.d4, self.d3, self.d2, self.d1 = Dec(96, ch[2], 64), Dec(64, ch[1], 48), Dec(48, ch[0], 32), Dec(32, 0, 16)
        else:  # late-only: no pyramid, light decoder from 1/4 resolution
            self.l1, self.l2 = DSConv(96, 48), DSConv(48, 24); self.l3 = DSConv(24, 16)
        self.head = nn.Conv2d(16, 1, 1)
    def _x(self, s, o):
        return torch.cat([t for t, k in ((s, ("both", "sar")), (o, ("both", "opt"))) if self.mod in k], 1)
    def forward(self, sb, ob, sa, oa):
        if self.use_early:
            fb, fa = self.enc(self._x(sb, ob)), self.enc(self._x(sa, oa))
            diffs = [torch.abs(a - b) for a, b in zip(fa, fb)]
            bott = diffs[3]
            if self.use_late:
                late = self.late(sa - sb, oa - ob)
                if late.shape[-2:] != bott.shape[-2:]: late = F.interpolate(late, size=bott.shape[-2:], mode="bilinear", align_corners=False)
                bott = bott + late
            d = self.d4(bott, diffs[2]); d = self.d3(d, diffs[1]); d = self.d2(d, diffs[0]); d = self.d1(d)
        else:
            x = self.late(sa - sb, oa - ob)
            x = self.l1(x); x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = self.l2(x); x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            d = self.l3(x)
            d = F.interpolate(d, size=sb.shape[-2:], mode="bilinear", align_corners=False) if d.shape[-2:] != sb.shape[-2:] else d
        return self.head(d)   # logits; apply sigmoid outside
