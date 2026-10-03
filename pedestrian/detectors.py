"""Єдиний інтерфейс для всіх детекторів пішоходів: кадр (BGR) -> прямокутники (N, 5) x1, y1, x2, y2, score.

Завдяки спільному інтерфейсу Detector решта коду (трекінг, оцінка, веб-інтерфейс) не залежить
від того, яка модель працює всередині — власна PedNet чи YOLO від Ultralytics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pedestrian.data import letterbox, normalize
from pedestrian.device import pick_device
from pedestrian.pednet import PedNet, count_parameters, decode


@dataclass
class ModelInfo:
    """Опис моделі для реєстру та інтерфейсу."""
    key: str           # короткий ідентифікатор (використовується в CLI та CSV)
    title: str         # назва для людини
    path: str          # шлях до файлу ваг
    kind: str          # "pednet" | "yolo" — який клас-обгортку використовувати
    origin: str        # "own" (власна) | "fine-tuned" (донавчена) | "pretrained" (з коробки)
    description: str


# Реєстр усіх моделей проєкту. Порядок визначає порядок у таблицях і списках інтерфейсу.
MODEL_ZOO: dict[str, ModelInfo] = {
    "pednet": ModelInfo("pednet", "PedNet (власна модель)", "models/pednet_best.pt", "pednet", "own",
                        "Anchor-free детектор центрів пішоходів: MobileNetV3 + власні FPN та голови, "
                        "навчений у проєкті на Caltech."),
    "yolo_caltech": ModelInfo("yolo_caltech", "YOLOv8n, донавчена на Caltech (12 епох)",
                              "models/yolo_caltech_v8n.pt", "yolo", "fine-tuned",
                              "YOLOv8n, донавчена на Caltech Pedestrian у цьому проєкті."),
    "yolo_old": ModelInfo("yolo_old", "YOLOv8n, донавчена (5 епох, попередня версія)",
                          "models/finetuned_best.pt", "yolo", "fine-tuned",
                          "Попередня версія fine-tune з першої ітерації лабораторної."),
    "yolo_coco": ModelInfo("yolo_coco", "YOLOv8n COCO (з коробки)", "models/yolov8n.pt", "yolo", "pretrained",
                           "Базова модель Ultralytics, навчена на COCO; використовується лише клас person."),
}

ORIGIN_LABELS = {"own": "власна", "fine-tuned": "донавчена", "pretrained": "з коробки"}


class Detector:
    """Базовий клас. Нащадки реалізують predict() та n_params()."""
    info: ModelInfo

    def predict(self, frame_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5) -> np.ndarray:
        """Повертає масив (N, 5): x1, y1, x2, y2, score у пікселях вхідного кадру."""
        raise NotImplementedError

    def n_params(self) -> int:
        raise NotImplementedError

    def size_mb(self) -> float:
        return Path(self.info.path).stat().st_size / 2 ** 20

    def timed_predict(self, frame_bgr, conf=0.25, iou=0.5):
        """predict() + час виконання в мілісекундах (включно з пре- та постобробкою)."""
        t0 = time.perf_counter()
        boxes = self.predict(frame_bgr, conf, iou)
        return boxes, (time.perf_counter() - t0) * 1000.0


class PedNetDetector(Detector):
    """Обгортка для власної моделі PedNet."""

    def __init__(self, info: ModelInfo, device: str | None = None):
        self.info = info
        self.device = pick_device(device)
        self.model = PedNet.load(info.path).to(self.device).eval()
        self.size = (self.model.cfg.input_w, self.model.cfg.input_h)

    @torch.no_grad()
    def predict(self, frame_bgr, conf=0.25, iou=0.5):
        # 1) приводимо кадр до розміру входу мережі зі збереженням пропорцій
        img, s, (px, py) = letterbox(frame_bgr, self.size)
        # 2) прямий прохід + декодування heatmap у прямокутники
        out = self.model(normalize(img)[None].to(self.device))
        d = decode(out, self.model.cfg.stride, conf=conf, nms_iou=iou)[0].numpy()
        # 3) переводимо координати назад у систему оригінального кадру (обернений letterbox)
        if len(d):
            d[:, [0, 2]] = (d[:, [0, 2]] - px) / s
            d[:, [1, 3]] = (d[:, [1, 3]] - py) / s
            h, w = frame_bgr.shape[:2]
            d[:, :4] = d[:, :4].clip(0, [w, h, w, h])
        return d.astype(np.float32)

    def n_params(self):
        return count_parameters(self.model)


class YOLODetector(Detector):
    """Обгортка для моделей Ultralytics YOLO (донавчених однокласових або COCO на 80 класів)."""

    def __init__(self, info: ModelInfo, device: str | None = None):
        from ultralytics import YOLO

        self.info = info
        self.device = pick_device(device)
        self.model = YOLO(info.path)
        names = self.model.names
        # модель COCO знає 80 класів — залишаємо лише "person"; однокласова модель — без фільтра
        self.classes = [k for k, v in names.items() if v == "person"] if len(names) > 1 else None

    def predict(self, frame_bgr, conf=0.25, iou=0.5):
        r = self.model.predict(frame_bgr, conf=conf, iou=iou, classes=self.classes,
                               device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 5), np.float32)
        return np.concatenate([r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()[:, None]], 1).astype(np.float32)

    def n_params(self):
        return sum(p.numel() for p in self.model.model.parameters())


def load_detector(key: str, device: str | None = None) -> Detector:
    """Створює детектор за ключем з MODEL_ZOO; помилка, якщо файл ваг ще не існує."""
    info = MODEL_ZOO[key]
    if not Path(info.path).exists():
        raise FileNotFoundError(f"Ваги моделі не знайдено: {info.path}")
    return PedNetDetector(info, device) if info.kind == "pednet" else YOLODetector(info, device)


def available_models() -> list[str]:
    """Ключі моделей, файли ваг яких уже є на диску."""
    return [k for k, m in MODEL_ZOO.items() if Path(m.path).exists()]
