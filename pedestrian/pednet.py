"""PedNet: an anchor-free single-class pedestrian detector built in this project.

Architecture
    backbone  MobileNetV3-Large (ImageNet weights, torchvision) - feature extractor only
    neck      top-down feature pyramid (FPN) with depthwise-separable blocks,
              fuses C2..C5 (strides 4..32) into one stride-4 map (small pedestrians!)
    head      three branches on the stride-4 map:
                heatmap (1)   - probability of a pedestrian centre
                size    (2)   - log(w), log(h) of the box in stride units
                offset  (2)   - sub-pixel shift of the centre
Loss        penalty-reduced focal loss on the heatmap + L1 on size and offset
Decoding    3x3 max-pool peak extraction -> top-K -> boxes -> NMS
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms


@dataclass
class PedNetConfig:
    neck_channels: int = 96
    head_channels: int = 64
    stride: int = 4
    input_w: int = 640
    input_h: int = 480
    pretrained_backbone: bool = True


def conv_bn_act(cin: int, cout: int, k: int = 1, s: int = 1, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.Hardswish(inplace=True),
    )


class DWSeparable(nn.Module):
    """Depthwise 3x3 + pointwise 1x1 with a residual connection."""

    def __init__(self, c: int):
        super().__init__()
        self.dw = conv_bn_act(c, c, 3, groups=c)
        self.pw = conv_bn_act(c, c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.dw(x))


class Backbone(nn.Module):
    """MobileNetV3-Large split into four stages C2 (s4), C3 (s8), C4 (s16), C5 (s32)."""

    out_channels = (24, 40, 112, 960)

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        f = torchvision.models.mobilenet_v3_large(weights=weights).features
        self.c2, self.c3, self.c4, self.c5 = f[:4], f[4:7], f[7:13], f[13:]

    def forward(self, x: torch.Tensor):
        c2 = self.c2(x)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.c5(c4)
        return c2, c3, c4, c5


class FPNNeck(nn.Module):
    """Top-down pathway: P5 -> P4 -> P3 -> P2, each level refined by a DW block."""

    def __init__(self, in_channels: tuple[int, ...], c: int):
        super().__init__()
        self.lateral = nn.ModuleList(conv_bn_act(ci, c, 1) for ci in in_channels)
        self.smooth = nn.ModuleList(DWSeparable(c) for _ in in_channels[:-1])

    def forward(self, feats):
        p = self.lateral[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            lat = self.lateral[i](feats[i])
            p = self.smooth[i](lat + F.interpolate(p, size=lat.shape[-2:], mode="nearest"))
        return p  # stride 4


class Head(nn.Module):
    def __init__(self, cin: int, c: int, cout: int, bias: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(cin, c, 3), DWSeparable(c))
        self.out = nn.Conv2d(c, cout, 1)
        nn.init.constant_(self.out.bias, bias)
        nn.init.normal_(self.out.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


class PedNet(nn.Module):
    def __init__(self, cfg: PedNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or PedNetConfig()
        self.backbone = Backbone(self.cfg.pretrained_backbone)
        self.neck = FPNNeck(Backbone.out_channels, self.cfg.neck_channels)
        c, hc = self.cfg.neck_channels, self.cfg.head_channels
        self.hm = Head(c, hc, 1, bias=-2.19)  # prior p=0.1 (focal loss init)
        self.size = Head(c, hc, 2, bias=1.0)
        self.off = Head(c, hc, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.neck(self.backbone(x))
        return {"hm": self.hm(p), "size": self.size(p), "off": self.off(p)}

    # ------------------------------------------------------------ checkpoints
    def save(self, path: str, **extra) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "PedNet":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = PedNetConfig(**{**ckpt["config"], "pretrained_backbone": False})
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model


# ---------------------------------------------------------------- loss
def focal_loss(pred_logits: torch.Tensor, gt: torch.Tensor, alpha: float = 2, beta: float = 4) -> torch.Tensor:
    """Penalty-reduced pixel-wise focal loss (CornerNet / CenterNet)."""
    pred = pred_logits.sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred ** alpha * (1 - gt) ** beta * neg
    num_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """feat (B, C, H, W), ind (B, K) -> (B, K, C)."""
    b, c = feat.shape[:2]
    feat = feat.view(b, c, -1).permute(0, 2, 1)
    return feat.gather(1, ind.unsqueeze(-1).expand(-1, -1, c))


def pednet_loss(out: dict, t: dict, w_size: float = 1.0, w_off: float = 1.0) -> dict[str, torch.Tensor]:
    l_hm = focal_loss(out["hm"], t["hm"])
    m = t["mask"].unsqueeze(-1)
    n = m.sum().clamp(min=1)
    l_size = (F.l1_loss(gather(out["size"], t["ind"]), t["size"], reduction="none") * m).sum() / n
    l_off = (F.l1_loss(gather(out["off"], t["ind"]), t["off"], reduction="none") * m).sum() / n
    total = l_hm + w_size * l_size + w_off * l_off
    return {"loss": total, "hm": l_hm.detach(), "size": l_size.detach(), "off": l_off.detach()}


# ---------------------------------------------------------------- decoding
@torch.no_grad()
def decode(out: dict, stride: int = 4, conf: float = 0.05, top_k: int = 100,
           nms_iou: float = 0.5) -> list[torch.Tensor]:
    """Return per-image tensors (N, 5): x1, y1, x2, y2, score in input-image pixels."""
    hm = out["hm"].sigmoid()
    peaks = (F.max_pool2d(hm, 3, 1, 1) == hm).float() * hm
    b, _, H, W = hm.shape
    scores, inds = peaks.view(b, -1).topk(min(top_k, H * W))
    size = gather(out["size"], inds).exp()
    off = gather(out["off"], inds)
    xs = (inds % W).float() + off[..., 0]
    ys = (inds // W).float() + off[..., 1]
    w, h = size[..., 0], size[..., 1]
    boxes = torch.stack([xs - w / 2, ys - h / 2, xs + w / 2, ys + h / 2], -1) * stride
    result = []
    for i in range(b):
        keep = scores[i] >= conf
        bx, sc = boxes[i][keep], scores[i][keep]
        if len(bx):
            k = nms(bx.float().cpu(), sc.float().cpu(), nms_iou)
            bx, sc = bx.cpu()[k], sc.cpu()[k]
        result.append(torch.cat([bx.cpu(), sc.cpu().unsqueeze(-1)], 1))
    return result


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
