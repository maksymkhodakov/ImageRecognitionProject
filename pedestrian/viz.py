"""Візуалізація: прямокутники, ідентифікатори треків, траєкторії руху."""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

ACCENT = (80, 200, 120)  # колір рамок детекцій (OpenCV використовує порядок BGR)


def track_color(track_id: int) -> tuple[int, int, int]:
    """Стабільний яскравий колір для кожного ID.

    Відтінок = дробова частина (id * золотий перетин): сусідні ID отримують контрастні кольори,
    а один і той самий пішохід — однаковий колір на всіх кадрах.
    """
    h = (track_id * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _label(img, text, x, y, color):
    """Підпис на кольоровій плашці над лівим верхнім кутом рамки (не виходить за верхній край)."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    y0 = max(th + 4, y)
    cv2.rectangle(img, (x, y0 - th - 4), (x + tw + 4, y0), color, -1)
    cv2.putText(img, text, (x + 2, y0 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, boxes: np.ndarray, color=ACCENT) -> np.ndarray:
    """Малює детекції (N, 5): x1, y1, x2, y2, score — рамка + впевненість. Оригінал не змінюється."""
    vis = frame.copy()
    for x1, y1, x2, y2, sc in boxes[:, :5]:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(vis, p1, p2, color, 2, cv2.LINE_AA)
        _label(vis, f"{sc:.2f}", p1[0], p1[1], color)
    return vis


def draw_tracks(frame: np.ndarray, tracks: np.ndarray, trails: dict | None = None) -> np.ndarray:
    """Малює треки (N, 6): x1, y1, x2, y2, score, id — кожен своїм кольором, з ID та траєкторією."""
    vis = frame.copy()
    for x1, y1, x2, y2, sc, tid in tracks:
        c = track_color(int(tid))
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), c, 2, cv2.LINE_AA)
        _label(vis, f"ID {int(tid)}  {sc:.2f}", int(x1), int(y1), c)
        if trails and int(tid) in trails and len(trails[int(tid)]) > 1:
            # траєкторія — ламана через точки «ніг» пішохода на попередніх кадрах
            pts = np.array(trails[int(tid)], np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, c, 2, cv2.LINE_AA)
    return vis
