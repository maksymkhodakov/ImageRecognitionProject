"""Datasets in YOLO format (Caltech Pedestrian, INRIA Person) and PedNet targets."""
from __future__ import annotations

import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

CALTECH_ROOT = Path("dataset/datasets")
INRIA_ROOT = Path("dataset/inria")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def list_images(root: Path, split: str) -> list[Path]:
    """All images of a split, sorted by name (= temporal order for Caltech)."""
    return sorted(p for p in (root / "images" / split).rglob("*") if p.suffix.lower() in IMG_EXTS)


def label_path(img_path: Path) -> Path:
    parts = list(img_path.parts)
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def read_boxes(img_path: Path, width: int, height: int) -> np.ndarray:
    """Read YOLO labels and return pixel boxes (N, 4) in xyxy format."""
    lp = label_path(img_path)
    if not lp.exists():
        return np.zeros((0, 4), np.float32)
    rows = [r.split() for r in lp.read_text().splitlines() if r.strip()]
    if not rows:
        return np.zeros((0, 4), np.float32)
    a = np.array(rows, dtype=np.float32)[:, 1:5]
    cx, cy, w, h = a[:, 0] * width, a[:, 1] * height, a[:, 2] * width, a[:, 3] * height
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1).clip(0, [width, height, width, height])


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize keeping aspect ratio and pad to size=(W, H). Returns image, scale, (pad_x, pad_y)."""
    tw, th = size
    h, w = img.shape[:2]
    s = min(tw / w, th / h)
    nw, nh = int(round(w * s)), int(round(h * s))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w, h) else img
    px, py = (tw - nw) // 2, (th - nh) // 2
    out = np.full((th, tw, 3), 114, np.uint8)
    out[py:py + nh, px:px + nw] = resized
    return out, s, (px, py)


def normalize(img_bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1).copy())


# ---------------------------------------------------------------- CenterNet targets
def gaussian_radius(h: float, w: float, min_overlap: float = 0.7) -> float:
    """Radius from CenterNet (Zhou et al., 2019): keeps IoU >= min_overlap."""
    a1, b1, c1 = 1, h + w, w * h * (1 - min_overlap) / (1 + min_overlap)
    r1 = (b1 + math.sqrt(b1 ** 2 - 4 * a1 * c1)) / 2
    a2, b2, c2 = 4, 2 * (h + w), (1 - min_overlap) * w * h
    r2 = (b2 + math.sqrt(b2 ** 2 - 4 * a2 * c2)) / 2
    a3, b3, c3 = 4 * min_overlap, -2 * min_overlap * (h + w), (min_overlap - 1) * w * h
    r3 = (b3 + math.sqrt(b3 ** 2 - 4 * a3 * c3)) / 2
    return min(r1, r2, r3)


def draw_gaussian(hm: np.ndarray, cx: int, cy: int, rx: float, ry: float) -> None:
    """Elliptic gaussian (pedestrians are tall and thin)."""
    H, W = hm.shape
    rx_i, ry_i = max(0, int(rx)), max(0, int(ry))
    sx, sy = (2 * rx_i + 1) / 6, (2 * ry_i + 1) / 6
    ys, xs = np.ogrid[-ry_i:ry_i + 1, -rx_i:rx_i + 1]
    g = np.exp(-(xs * xs) / (2 * sx * sx) - (ys * ys) / (2 * sy * sy))
    l, r = min(cx, rx_i), min(W - cx, rx_i + 1)
    t, b = min(cy, ry_i), min(H - cy, ry_i + 1)
    if r <= 0 or b <= 0 or l < 0 or t < 0:
        return
    patch = hm[cy - t:cy + b, cx - l:cx + r]
    np.maximum(patch, g[ry_i - t:ry_i + b, rx_i - l:rx_i + r], out=patch)


def build_targets(boxes: np.ndarray, out_hw: tuple[int, int], stride: int, max_objs: int = 128) -> dict:
    """Heatmap, log-size and sub-pixel offset targets for PedNet."""
    H, W = out_hw
    hm = np.zeros((H, W), np.float32)
    size = np.zeros((max_objs, 2), np.float32)
    off = np.zeros((max_objs, 2), np.float32)
    ind = np.zeros((max_objs,), np.int64)
    mask = np.zeros((max_objs,), np.float32)
    # biggest boxes first so that small ones win the heatmap peaks
    order = np.argsort(-(boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])) if len(boxes) else []
    k = 0
    for i in order:
        x1, y1, x2, y2 = boxes[i] / stride
        w, h = x2 - x1, y2 - y1
        if w <= 0.5 or h <= 0.5 or k >= max_objs:
            continue
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ix, iy = min(int(cx), W - 1), min(int(cy), H - 1)
        r = max(0.0, gaussian_radius(h, w))
        draw_gaussian(hm, ix, iy, max(0.0, r * min(1.0, w / h) ** 0.5), max(0.0, r))
        size[k] = (math.log(w), math.log(h))
        off[k] = (cx - ix, cy - iy)
        ind[k] = iy * W + ix
        mask[k] = 1.0
        k += 1
    return {"hm": hm[None], "size": size, "off": off, "ind": ind, "mask": mask}


# ---------------------------------------------------------------- dataset
class PedestrianDataset(Dataset):
    """YOLO-format pedestrian dataset producing PedNet training samples."""

    def __init__(self, images: list[Path], input_size=(640, 480), stride: int = 4,
                 augment: bool = False, min_box_h: float = 8.0):
        self.images = images
        self.size = input_size
        self.stride = stride
        self.augment = augment
        self.min_box_h = min_box_h

    def __len__(self) -> int:
        return len(self.images)

    def load(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(str(self.images[i]))
        h, w = img.shape[:2]
        return img, read_boxes(self.images[i], w, h)

    def _augment(self, img: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = img.shape[:2]
        # random scale + crop/pad (zoom in helps small pedestrians)
        s = random.uniform(0.75, 1.35)
        nw, nh = int(w * s), int(h * s)
        img = cv2.resize(img, (nw, nh))
        boxes = boxes * s
        canvas = np.full((h, w, 3), 114, np.uint8)
        ox = random.randint(min(0, w - nw), max(0, w - nw))
        oy = random.randint(min(0, h - nh), max(0, h - nh))
        sx0, sy0 = max(0, -ox), max(0, -oy)
        dx0, dy0 = max(0, ox), max(0, oy)
        cw, ch = min(nw - sx0, w - dx0), min(nh - sy0, h - dy0)
        canvas[dy0:dy0 + ch, dx0:dx0 + cw] = img[sy0:sy0 + ch, sx0:sx0 + cw]
        img = canvas
        if len(boxes):
            boxes = boxes + np.array([ox, oy, ox, oy], np.float32)
            area0 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes.clip(0, [w, h, w, h])
            area1 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            boxes = boxes[area1 > 0.4 * np.maximum(area0, 1e-6)]
        # horizontal flip
        if random.random() < 0.5:
            img = img[:, ::-1].copy()
            if len(boxes):
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        # colour jitter in HSV
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
            boxes = boxes * s + np.array([px, py, px, py], np.float32)
            boxes = boxes[(boxes[:, 3] - boxes[:, 1]) >= self.min_box_h * s]
        W, H = self.size
        t = build_targets(boxes.astype(np.float32), (H // self.stride, W // self.stride), self.stride)
        t = {k: torch.from_numpy(v) for k, v in t.items()}
        return normalize(img), t, torch.from_numpy(boxes.astype(np.float32))


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    targets = {k: torch.stack([b[1][k] for b in batch]) for k in batch[0][1]}
    boxes = [b[2] for b in batch]
    return imgs, targets, boxes
