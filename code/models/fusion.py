"""Fusion blocks. Branches that an ablation removes are really deleted, not masked."""
import torch, torch.nn as nn

class CrossModalAttention(nn.Module):
    """Squeeze-and-excitation channel attention (reduction 8)."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True),
                                nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.shape
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)

class DSConv(nn.Module):
    def __init__(self, i, o, stride=1):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(i, i, 3, stride, 1, groups=i, bias=False), nn.BatchNorm2d(i), nn.ReLU6(True),
                               nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU6(True))
    def forward(self, x): return self.b(x)

class LateBranch(nn.Module):
    """Per-modality DS sub-encoders on the pre->post difference, then optional channel attention.
    If one modality has 0 channels its encoder does not exist."""
    def __init__(self, sar_ch=2, opt_ch=8, out_ch=96, use_cma=True):
        super().__init__()
        n = (sar_ch > 0) + (opt_ch > 0); w = out_ch // n
        mk = lambda c: nn.Sequential(DSConv(c, 32, 2), DSConv(32, w, 2))
        self.sar = mk(sar_ch) if sar_ch > 0 else None
        self.opt = mk(opt_ch) if opt_ch > 0 else None
        self.att = CrossModalAttention(w * n) if use_cma else nn.Identity()
        self.out_ch = w * n
    def forward(self, ds, do):
        f = [m(x) for m, x in ((self.sar, ds), (self.opt, do)) if m is not None]
        return self.att(torch.cat(f, 1) if len(f) > 1 else f[0])
