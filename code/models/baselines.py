import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.models import resnet34
from .fusion import DSConv

class _Conv(nn.Sequential):
    def __init__(self, i, o): super().__init__(nn.Conv2d(i, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                                                nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

class UNetR34(nn.Module):
    """Early-concatenation U-Net with ResNet34 encoder: input = [pre | post] stacked on channels.
    inputs: modality 'opt' (8+8 ch), 'sar' (2+2) or 'both' (10+10)."""
    def __init__(self, modalities="both", pretrained=False):
        super().__init__()
        self.mod = modalities; c = {"both": 20, "opt": 16, "sar": 4}[modalities]
        r = resnet34(weights="DEFAULT" if pretrained else None)
        conv1 = nn.Conv2d(c, 64, 7, 2, 3, bias=False)
        if pretrained:
            with torch.no_grad(): conv1.weight.copy_(r.conv1.weight.mean(1, keepdim=True).repeat(1, c, 1, 1))
        self.l0 = nn.Sequential(conv1, r.bn1, r.relu)   # 1/2, 64
        self.pool = r.maxpool; self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.u4 = _Conv(512 + 256, 256); self.u3 = _Conv(256 + 128, 128); self.u2 = _Conv(128 + 64, 64); self.u1 = _Conv(64 + 64, 32)
        self.head = nn.Conv2d(32, 1, 1)
    def forward(self, sb, ob, sa, oa):
        xs = {"both": [sb, ob, sa, oa], "opt": [ob, oa], "sar": [sb, sa]}[self.mod]
        x = torch.cat(xs, 1); H, W = x.shape[-2:]
        e0 = self.l0(x); e1 = self.l1(self.pool(e0)); e2 = self.l2(e1); e3 = self.l3(e2); e4 = self.l4(e3)
        up = lambda t, s: torch.cat([F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1)
        d = self.u4(up(e4, e3)); d = self.u3(up(d, e2)); d = self.u2(up(d, e1)); d = self.u1(up(d, e0))
        return self.head(F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False))

class PlainUNet(nn.Module):
    """Small U-Net trained from scratch on stacked [pre|post] input (no pretrained weights)."""
    def __init__(self, modalities="both"):
        super().__init__()
        self.mod = modalities; c = {"both": 20, "opt": 16, "sar": 4}[modalities]
        self.e1, self.e2, self.e3, self.e4 = _Conv(c, 32), _Conv(32, 64), _Conv(64, 128), _Conv(128, 256)
        self.d3, self.d2, self.d1 = _Conv(256 + 128, 128), _Conv(128 + 64, 64), _Conv(64 + 32, 32)
        self.head = nn.Conv2d(32, 1, 1)
    def forward(self, sb, ob, sa, oa):
        xs = {"both": [sb, ob, sa, oa], "opt": [ob, oa], "sar": [sb, sa]}[self.mod]
        x = torch.cat(xs, 1)
        e1 = self.e1(x); e2 = self.e2(F.max_pool2d(e1, 2)); e3 = self.e3(F.max_pool2d(e2, 2)); e4 = self.e4(F.max_pool2d(e3, 2))
        up = lambda t, s: torch.cat([F.interpolate(t, size=s.shape[-2:], mode="bilinear", align_corners=False), s], 1)
        d = self.d3(up(e4, e3)); d = self.d2(up(d, e2)); d = self.d1(up(d, e1)); return self.head(d)
