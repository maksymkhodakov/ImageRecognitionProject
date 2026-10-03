"""Робота з даними: датасети у форматі YOLO (Caltech Pedestrian, INRIA Person) та цілі для PedNet.

Формат YOLO: для кожного зображення images/.../name.png існує файл labels/.../name.txt,
де кожен рядок — один об'єкт: "class x_center y_center width height", координати
нормовані на розміри зображення (0..1). У нас лише один клас — 0 (person).
"""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CALTECH_ROOT = Path("dataset/datasets")  # images/{train,val,test} + labels/{train,val,test}
INRIA_ROOT = Path("dataset/inria")       # створюється скриптом prepare_inria.py
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# Середнє та стандартне відхилення ImageNet: backbone MobileNetV3 навчений саме з такою нормалізацією.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_images(root: Path, split: str) -> list[Path]:
    """Усі зображення вибірки, відсортовані за іменем.

    Для Caltech імена мають вигляд setXX_VYYY_FFFF, тому сортування за іменем = часовий
    порядок кадрів усередині кожного відео (це важливо для трекінгу).
    """
    return sorted(p for p in (root / "images" / split).rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path(img_path: Path) -> Path:
    """Шлях до файлу розмітки: остання частина шляху "images" замінюється на "labels", розширення — на .txt."""
    parts = list(img_path.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(img_path: Path, width: int, height: int) -> np.ndarray:
    """Читає YOLO-розмітку й повертає прямокутники в пікселях: масив (N, 4) у форматі x1, y1, x2, y2."""
    lp = label_path(img_path)
    if not lp.exists():
        return np.zeros((0, 4), np.float32)
    rows = [r.split() for r in lp.read_text().splitlines() if r.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.array(rows, dtype=np.float32)[:, 1:5]  # відкидаємо номер класу
    # денормалізація: частки від розміру зображення -> пікселі
    cx, cy, w, h = a[:, 0] * width, a[:, 1] * height, a[:, 2] * width, a[:, 3] * height
    # (центр, розмір) -> (лівий верхній, правий нижній кут), обрізання межами кадру
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).clip(0, [width, height, width, height])


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Масштабує зображення зі збереженням пропорцій і доповнює сірими смугами до size=(W, H).

    Повертає (зображення, масштаб s, (pad_x, pad_y)). Ці значення потрібні, щоб перевести
    передбачені прямокутники назад у координати оригінального кадру: x_orig = (x - pad_x) / s.
    Пропорції не спотворюються — інакше пішоходи ставали б «товстішими» чи «тоншими».
    """
    tw, th = size
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    px, py = (tw - nw) // 2, (th - nh) // 2
    out = np.full((th, tw, 3), 114, np.uint8)  # 114 — нейтральний сірий (як у YOLO)
    out[py:py + nh, px:px + nw] = resized
    return out, s, (px, py)


def normalize(img_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 (H, W, 3) -> нормалізований тензор RGB float32 (3, H, W)."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy())


# ---------------------------------------------------------------- цілі в стилі CenterNet
def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """Радіус гаусіана з CenterNet (Zhou et al., 2019).

    Шукається найбільший радіус r такий, що прямокутник, центр якого зсунуто на r комірок,
    все ще має IoU >= min_overlap з істинним. Розглядаються три випадки взаємного
    розташування (два кути всередині / зовні / змішано) — кожен дає квадратне рівняння,
    з якого береться корінь; підсумковий радіус — мінімальний з трьох.
    """
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(hm: np.ndarray, cx: int, cy: int, rx: float, ry: float) -> None:
    """Малює на карті hm еліптичний гаусіан з центром (cx, cy) і радіусами rx, ry (на місці).

    Еліптичний, бо пішоходи високі й вузькі: по горизонталі «дозволений» зсув центру менший.
    sigma = діаметр / 6, тож на межі радіуса значення гаусіана ≈ exp(-4.5) ≈ 0.01.
    Перекриття з іншими пішоходами береться за максимумом, а не сумою.
    """
    H, W = hm.shape
    rx_i, ry_i = max(0, int(rx)), max(0, int(ry))
    sx, sy = (2 * rx_i + 1) / 6, (2 * ry_i + 1) / 6
    ys, xs = np.ogrid[-ry_i:ry_i + 1, -rx_i:rx_i + 1]
    g = np.exp(-(xs * xs) / (2 * sx * sx) - (ys * ys) / (2 * sy * sy))
    # обрізаємо гаусіан межами карти (пішохід біля краю кадру)
    l, r = min(cx, rx_i), min(W - cx, rx_i + 1)
    t, b = min(cy, ry_i), min(H - cy, ry_i + 1)
    if r <= 0 or b <= 0 or l < 0 or t < 0:
        return
    patch = hm[cy - t:cy + b, cx - l:cx + r]
    np.maximum(patch, g[ry_i - t:ry_i + b, rx_i - l:rx_i + r], out=patch)


def build_targets(boxes: np.ndarray, out_hw: tuple[int, int], stride: int, max_objs: int = 128) -> dict:
    """Будує цілі для навчання PedNet з прямокутників (у пікселях входу мережі).

    Повертає:
      hm   (1, H, W)        — цільова карта центрів (1.0 у точному центрі, гаусіан навколо);
      size (max_objs, 2)    — log(w), log(h) у комірках (логарифм вирівнює масштаби
                              маленьких і великих пішоходів для L1-втрати);
      off  (max_objs, 2)    — дробова частина центру (центр 37.6 -> комірка 37, зсув 0.6);
      ind  (max_objs,)      — лінійний індекс комірки центру y * W + x;
      mask (max_objs,)      — 1 для реальних об'єктів, 0 для порожніх слотів.
    Фіксований розмір max_objs потрібен, щоб зібрати батч з кадрів з різною кількістю людей.
    """
    H, W = out_hw
    hm = np.zeros((H, W), np.float32)
    size = np.zeros((max_objs, 2), np.float32)
    off = np.zeros((max_objs, 2), np.float32)
    ind = np.zeros((max_objs,), np.int64)
    mask = np.zeros((max_objs,), np.float32)
    # Спершу великі прямокутники: якщо центри збігаються, ціль розміру перезапише менший об'єкт.
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) if len(boxes) else []
    k = 0
    for i in order:
        x1, y1, x2, y2 = boxes[i] / stride  # пікселі -> комірки вихідної карти
        w, h = x2 - x1, y2 - y1
        if w <= 0.5 or h <= 0.5 or k >= max_objs:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ix, iy = min(int(cx), W - 1), min(int(cy), H - 1)
        r = max(0.0, gaussian_radius(h, w))
        # горизонтальний радіус зменшується пропорційно sqrt(w/h); мінімум 1 комірка,
        # щоб навіть дрібні пішоходи мали «пляму», а не одну точку (стабільніше навчання)
        draw_gaussian(hm, ix, iy, max(1.0, r * min(1.0, w / h) ** 0.5), max(1.0, r))
        size[k] = (math.log(w), math.log(h))
        off[k] = (cx - ix, cy - iy)
        ind[k] = iy * W + ix
        mask[k] = 1.0
        k += 1
    return {"hm": hm[None], "size": size, "off": off, "ind": ind, "mask": mask}


# ---------------------------------------------------------------- датасет
class PedestrianDataset(Dataset):
    """Датасет у форматі YOLO, що повертає приклади для навчання PedNet.

    Кожен елемент: (нормалізований тензор зображення, словник цілей, прямокутники у пікселях).
    Прямокутники потрібні для валідації (обчислення AP), цілі — для функції втрат.
    """

    def __init__(self, images: list[Path], input_size=(640, 480), stride: int = 4,
                 augment: bool = False, min_box_h: float = 8.0):
        self.images = images
        self.size = input_size      # (W, H) входу мережі
        self.stride = stride
        self.augment = augment      # аугментації лише для навчальної вибірки
        self.min_box_h = min_box_h  # пішоходи нижчі за цю висоту (px) ігноруються

    def __len__(self) -> int:
        return len(self.images)

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(self.images[i]))
        h, w = img.shape[:2]
        return img, read_boxes(self.images[i], w, h)

    def _augment(self, img: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Випадкові перетворення, що імітують різноманітність реальних умов зйомки."""
        h, w = img.shape[:2]
        # 1) Випадковий масштаб 0.75–1.35 з обрізанням або доповненням до вихідного розміру:
        #    пішоходи з'являються в різних розмірах і різних місцях кадру.
        s = random.uniform(0.75, 1.35)
        nw, nh = int(w * s), int(h * s)
        img = cv2.resize(img, (nw, nh))
        boxes = boxes * s
        canvas = np.full((h, w, 3), 114, np.uint8)
        # ox, oy — зсув масштабованого зображення на полотні (від'ємний = обрізаємо зліва/зверху)
        ox = random.randint(min(0, w - nw), max(0, w - nw))
        oy = random.randint(min(0, h - nh), max(0, h - nh))
        sx0, sy0 = max(0, -ox), max(0, -oy)   # звідки копіювати з масштабованого зображення
        dx0, dy0 = max(0, ox), max(0, oy)     # куди копіювати на полотні
        cw, ch = min(nw - sx0, w - dx0), min(nh - sy0, h - dy0)
        canvas[dy0:dy0 + ch, dx0:dx0 + cw] = img[sy0:sy0 + ch, sx0:sx0 + cw]
        img = canvas
        if len(boxes):
            boxes = boxes + np.array([ox, oy, ox, oy], np.float32)
            area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes.clip(0, [w, h, w, h])
            area1 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            # пішохід, від якого в кадрі лишилося менше 40% площі, вважається «втраченим»
            boxes = boxes[area1 > 0.4 * np.maximum(area0, 1e-6)]
        # 2) Горизонтальне віддзеркалення (пішоходи симетричні зліва-направо).
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        # 3) Зміна насиченості та яскравості в просторі HSV (різне освітлення, погода, камера).
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= random.uniform(0.6, 1.4)
        hsv[..., 2] *= random.uniform(0.6, 1.4)
        img = cv2.cvtColor(hsv.clip(0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return img, boxes

    def __getitem__(self, i: int):
        img, boxes = self.load(i)
        if self.augment:
            img, boxes = self._augment(img, boxes)
        img, s, (px, py) = letterbox(img, self.size)
        if len(boxes):
            # переводимо прямокутники в координати входу мережі (після letterbox)
            boxes = boxes * s + np.array([px, py, px, py], np.float32)
            boxes = boxes[(boxes[:, 3] - boxes[:, 1]) >= self.min_box_h * s]
        W, H = self.size
        t = build_targets(boxes.astype(np.float32), (H // self.stride, W // self.stride), self.stride)
        t = {k: torch.from_numpy(v) for k, v in t.items()}
        return normalize(img), t, torch.from_numpy(boxes.astype(np.float32))


def collate(batch):
    """Збирає батч: зображення та цілі — у тензори; прямокутники — у список (різна кількість на кадр)."""
    imgs = torch.stack([b[0] for b in batch])
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    boxes = [b[2] for b in batch]
    return imgs, targets, boxes
