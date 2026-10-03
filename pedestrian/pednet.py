"""PedNet — власний anchor-free однокласовий детектор пішоходів, розроблений у цьому проєкті.

Ідея (у стилі CenterNet, «Objects as Points»): кожен пішохід описується однією точкою — центром
свого прямокутника — та двома атрибутами: розміром (w, h) і субпіксельним зсувом центру.
Мережа не використовує якорі (anchor boxes) і не потребує складного зіставлення якорів з об'єктами.

Архітектура
    backbone  MobileNetV3-Large (ваги ImageNet, torchvision) — лише як екстрактор ознак;
              класифікаційна частина відкинута.
    neck      власна піраміда ознак (FPN) згори вниз на depthwise-separable згортках:
              зливає карти C2..C5 (кроки 4..32) в одну карту з кроком 4 — це важливо
              для точної локалізації вузьких (30–40 px) пішоходів.
    head      три гілки на карті з кроком 4:
                heatmap (1 канал) — ймовірність, що в цій комірці знаходиться центр пішохода;
                size    (2 канали) — log(w), log(h) прямокутника в одиницях кроку;
                offset  (2 канали) — субпіксельний зсув центру (компенсує дискретизацію).
Функція втрат: focal loss зі зниженим штрафом на heatmap + L1 на розмір та зсув.
Декодування: пошук локальних максимумів (max-pool 3x3) -> top-K -> прямокутники -> NMS.

Розміри тензорів для кадру 640x480: вхід (B, 3, 480, 640) -> виходи (B, C, 120, 160).
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
    """Гіперпараметри архітектури (зберігаються разом із вагами в чекпойнті)."""
    neck_channels: int = 96         # кількість каналів у всіх рівнях FPN
    head_channels: int = 64         # кількість каналів усередині голів
    stride: int = 4                 # у скільки разів вихідна карта менша за вхід
    input_w: int = 640              # розмір входу мережі (рідна роздільність Caltech)
    input_h: int = 480
    pretrained_backbone: bool = True  # завантажувати ваги ImageNet для MobileNetV3


def conv_bn_act(cin: int, cout: int, k: int = 1, s: int = 1, groups: int = 1) -> nn.Sequential:
    """Базовий блок: згортка k x k -> BatchNorm -> Hardswish.

    bias=False, бо BatchNorm одразу після згортки має власний зсув (beta).
    groups=cin перетворює згортку на depthwise (кожен канал обробляється окремо).
    """
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
        nn.BatchNorm2d(cout),
        nn.Hardswish(inplace=True),  # та сама активація, що й у MobileNetV3
    )


class DWSeparable(nn.Module):
    """Depthwise-separable блок із залишковим з'єднанням: x + PW1x1(DW3x3(x)).

    Розкладання звичайної згортки 3x3 на depthwise 3x3 (просторова частина) і pointwise 1x1
    (змішування каналів) зменшує кількість параметрів і обчислень приблизно в 8–9 разів.
    Залишкове з'єднання (x + ...) полегшує проходження градієнта.
    """

    def __init__(self, c: int):
        super().__init__()
        self.dw = conv_bn_act(c, c, 3, groups=c)  # просторова згортка окремо для кожного каналу
        self.pw = conv_bn_act(c, c, 1)            # змішування каналів

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pw(self.dw(x))


class Backbone(nn.Module):
    """MobileNetV3-Large, розділена на чотири стадії C2 (крок 4), C3 (8), C4 (16), C5 (32).

    Межі стадій [:4], [4:7], [7:13], [13:] обрано за місцями, де MobileNetV3 зменшує
    просторову роздільність (див. features[i] у torchvision).
    """

    out_channels = (24, 40, 112, 960)  # кількість каналів C2, C3, C4, C5

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        f = torchvision.models.mobilenet_v3_large(weights=weights).features
        self.c2, self.c3, self.c4, self.c5 = f[:4], f[4:7], f[7:13], f[13:]

    def forward(self, x: torch.Tensor):
        c2 = self.c2(x)   # (B, 24, H/4,  W/4)  — дрібні деталі, точна локалізація
        c3 = self.c3(c2)  # (B, 40, H/8,  W/8)
        c4 = self.c4(c3)  # (B, 112, H/16, W/16)
        c5 = self.c5(c4)  # (B, 960, H/32, W/32) — найбільш «семантичні» ознаки
        return c2, c3, c4, c5


class FPNNeck(nn.Module):
    """Власна піраміда ознак згори вниз: P5 -> P4 -> P3 -> P2.

    На кожному рівні: бічна проєкція 1x1 до спільної кількості каналів, додавання
    збільшеної вдвічі карти з глибшого рівня та уточнення DW-блоком. Результат — одна
    карта з кроком 4, що поєднує семантику глибоких шарів з точністю неглибоких.
    """

    def __init__(self, in_channels: tuple[int, ...], c: int):
        super().__init__()
        self.lateral = nn.ModuleList(conv_bn_act(ci, c, 1) for ci in in_channels)
        self.smooth = nn.ModuleList(DWSeparable(c) for _ in in_channels[:-1])

    def forward(self, feats):
        p = self.lateral[-1](feats[-1])  # починаємо з найглибшого рівня C5
        for i in range(len(feats) - 2, -1, -1):  # C4, C3, C2
            lat = self.lateral[i](feats[i])
            # nearest-збільшення до розміру поточного рівня + поелементне додавання
            p = self.smooth[i](lat + F.interpolate(p, size=lat.shape[-2:], mode="nearest"))
        return p  # крок 4


class Head(nn.Module):
    """Голова передбачення: conv3x3 -> DW-блок -> conv1x1 з cout виходами."""

    def __init__(self, cin: int, c: int, cout: int, bias: float = 0.0):
        super().__init__()
        self.body = nn.Sequential(conv_bn_act(cin, c, 3), DWSeparable(c))
        self.out = nn.Conv2d(c, cout, 1)
        # Малі початкові ваги + заданий зсув: на старті навчання вихід ≈ bias для всіх комірок.
        nn.init.constant_(self.out.bias, bias)
        nn.init.normal_(self.out.weight, std=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(x))


class PedNet(nn.Module):
    """Повна модель: backbone -> FPN -> три голови. Повертає «сирі» карти (логіти)."""

    def __init__(self, cfg: PedNetConfig | None = None):
        super().__init__()
        self.cfg = cfg or PedNetConfig()
        self.backbone = Backbone(self.cfg.pretrained_backbone)
        self.neck = FPNNeck(Backbone.out_channels, self.cfg.neck_channels)
        c, hc = self.cfg.neck_channels, self.cfg.head_channels
        # bias = -2.19 = log(0.1 / 0.9): на старті sigmoid(heatmap) ≈ 0.1 скрізь. Без цього
        # величезна кількість «фонових» комірок дала б дуже великий початковий loss (як у RetinaNet).
        self.hm = Head(c, hc, 1, bias=-2.19)
        self.size = Head(c, hc, 2, bias=1.0)  # exp(1) ≈ 2.7 комірки — розумний стартовий розмір
        self.off = Head(c, hc, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        p = self.neck(self.backbone(x))
        return {"hm": self.hm(p), "size": self.size(p), "off": self.off(p)}

    # ------------------------------------------------------------ чекпойнти
    def save(self, path: str, **extra) -> None:
        """Зберігає конфігурацію + ваги (та довільні додаткові поля, напр. номер епохи)."""
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "PedNet":
        """Відновлює модель з чекпойнта. Ваги ImageNet не завантажуються — їх замінять збережені."""
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        cfg = PedNetConfig(**{**ckpt["config"], "pretrained_backbone": False})
        model = cls(cfg)
        model.load_state_dict(ckpt["state_dict"])
        return model


# ---------------------------------------------------------------- функція втрат
def focal_loss(pred_logits: torch.Tensor, gt: torch.Tensor, alpha: float = 2, beta: float = 4) -> torch.Tensor:
    """Pixel-wise focal loss зі зниженим штрафом (CornerNet / CenterNet).

    Позитивні комірки (gt == 1, точний центр):  -(1 - p)^alpha * log(p)
    Негативні комірки:                           -(1 - gt)^beta * p^alpha * log(1 - p)
    Множник (1 - gt)^beta зменшує штраф поблизу центрів, де gt — значення гаусіана:
    сусідня з центром комірка — «майже правильна» відповідь. Множники (1-p)^alpha і p^alpha
    зменшують вагу легких прикладів, щоб мільйони тривіальних фонових комірок не домінували.
    Нормалізація — на кількість пішоходів (позитивних комірок).
    """
    pred = pred_logits.sigmoid().clamp(1e-4, 1 - 1e-4)  # clamp захищає від log(0)
    pos = gt.eq(1).float()
    neg = 1 - pos
    pos_loss = torch.log(pred) * (1 - pred) ** alpha * pos
    neg_loss = torch.log(1 - pred) * pred ** alpha * (1 - gt) ** beta * neg
    num_pos = pos.sum().clamp(min=1)
    return -(pos_loss.sum() + neg_loss.sum()) / num_pos


def gather(feat: torch.Tensor, ind: torch.Tensor) -> torch.Tensor:
    """Вибирає значення карти в заданих комірках: feat (B, C, H, W), ind (B, K) -> (B, K, C).

    ind — лінійний індекс комірки (y * W + x). Так розмір і зсув рахуються лише в центрах пішоходів.
    """
    b, c = feat.shape[:2]
    feat = feat.view(b, c, -1).permute(0, 2, 1)  # (B, H*W, C)
    return feat.gather(1, ind.unsqueeze(-1).expand(-1, -1, c))


def pednet_loss(out: dict, t: dict, w_size: float = 1.0, w_off: float = 1.0) -> dict[str, torch.Tensor]:
    """Загальна втрата L = L_hm + w_size * L_size + w_off * L_off.

    out — виходи мережі; t — цілі з PedestrianDataset (hm, size, off, ind, mask).
    mask відмічає реальні об'єкти серед max_objs слотів (решта — заповнення нулями).
    Повертає словник: "loss" для backward та окремі компоненти для логування.
    """
    l_hm = focal_loss(out["hm"], t["hm"])
    m = t["mask"].unsqueeze(-1)
    n = m.sum().clamp(min=1)
    l_size = (F.l1_loss(gather(out["size"], t["ind"]), t["size"], reduction="none") * m).sum() / n
    l_off = (F.l1_loss(gather(out["off"], t["ind"]), t["off"], reduction="none") * m).sum() / n
    total = l_hm + w_size * l_size + w_off * l_off
    return {"loss": total, "hm": l_hm.detach(), "size": l_size.detach(), "off": l_off.detach()}


# ---------------------------------------------------------------- декодування
@torch.no_grad()
def decode(out: dict, stride: int = 4, conf: float = 0.05, top_k: int = 100,
           nms_iou: float = 0.5) -> list[torch.Tensor]:
    """Перетворює виходи мережі на прямокутники.

    1. sigmoid -> ймовірності центрів;
    2. «піки»: комірка лишається, лише якщо вона максимум у своєму околі 3x3 (дешевий аналог NMS);
    3. top_k найсильніших піків з ймовірністю >= conf;
    4. центр = (x + offset_x, y + offset_y), розмір = exp(size), усе множиться на stride;
    5. додатковий NMS прибирає дублікати перекритих пішоходів.
    Повертає список (по одному на зображення) тензорів (N, 5): x1, y1, x2, y2, score у пікселях входу.
    """
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
            # NMS виконуємо на CPU: на MPS ця операція підтримується не в усіх версіях torchvision
            k = nms(bx.float().cpu(), sc.float().cpu(), nms_iou)
            bx, sc = bx.cpu()[k], sc.cpu()[k]
        result.append(torch.cat([bx.cpu(), sc.cpu().unsqueeze(-1)], 1))
    return result


def count_parameters(model: nn.Module) -> int:
    """Загальна кількість параметрів моделі."""
    return sum(p.numel() for p in model.parameters())
