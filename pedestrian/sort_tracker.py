"""Трекер SORT (Bewley et al., 2016), реалізований з нуля.

Парадигма tracking-by-detection: детектор незалежно знаходить пішоходів на кожному кадрі,
а трекер пов'язує ці детекції між кадрами в траєкторії з постійними ідентифікаторами (ID).

Кожен трек має власний фільтр Калмана з моделлю постійної швидкості над станом
    [cx, cy, s, r, vcx, vcy, vs]
    cx, cy — центр прямокутника; s — площа; r — відношення сторін w/h (вважається сталим);
    vcx, vcy, vs — швидкості зміни центру та площі (оцінюються фільтром, не спостерігаються).
Детекції зіставляються з прогнозами треків за IoU угорським алгоритмом (оптимальне призначення).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from pedestrian.metrics import box_iou


def xyxy_to_z(b: np.ndarray) -> np.ndarray:
    """Прямокутник x1, y1, x2, y2 -> вектор вимірювання z = [cx, cy, s, r]."""
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)], np.float64)


def x_to_xyxy(x: np.ndarray) -> np.ndarray:
    """Стан фільтра [cx, cy, s, r, ...] -> прямокутник x1, y1, x2, y2 (w = sqrt(s*r), h = s/w)."""
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxFilter:
    """Лінійний фільтр Калмана з постійною швидкістю для cx, cy та площі.

    F — матриця переходу (x_{t+1} = F x_t): позиція += швидкість.
    H — матриця спостереження: вимірюємо лише перші 4 компоненти (cx, cy, s, r).
    R — шум вимірювань (площа й пропорції детектора шумніші за центр).
    P — коваріація похибки стану: велика початкова невизначеність швидкостей (1e4),
        бо з першої детекції швидкість невідома.
    Q — шум процесу (наскільки рух пішохода може відхилятися від рівномірного).
    """

    def __init__(self, box: np.ndarray):
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0])
        self.P = np.diag([10.0, 10.0, 10.0, 10.0, 1e4, 1e4, 1e4])
        self.Q = np.diag([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 1e-4])
        self.x = np.zeros(7)
        self.x[:4] = xyxy_to_z(box)

    def predict(self) -> np.ndarray:
        """Крок прогнозу: де буде пішохід на наступному кадрі. Повертає прогнозований прямокутник."""
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0  # площа не може стати від'ємною
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_xyxy(self.x)

    def update(self, box: np.ndarray) -> None:
        """Крок корекції: уточнюємо стан за новою детекцією.

        y — інновація (різниця між виміром і прогнозом), S — її коваріація,
        K — коефіцієнт Калмана: наскільки довіряти виміру порівняно з прогнозом.
        """
        z = xyxy_to_z(box)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    @property
    def box(self) -> np.ndarray:
        return x_to_xyxy(self.x)


@dataclass
class Track:
    """Один відстежуваний пішохід."""
    id: int                   # постійний ідентифікатор
    kf: KalmanBoxFilter
    score: float              # впевненість останньої детекції
    hits: int = 1             # скільки разів трек підтверджувався детекцією
    hit_streak: int = 1       # скільки кадрів поспіль підтверджувався
    age: int = 0              # скільки кадрів існує
    time_since_update: int = 0  # скільки кадрів без детекції (живе лише за прогнозом)
    history: list = field(default_factory=list)  # точки (центр низу прямокутника) для малювання траєкторії


class SortTracker:
    """SORT: прогноз Калмана + угорський алгоритм за IoU.

    max_age       — скільки кадрів трек може жити без детекції (переживає короткі перекриття);
    min_hits      — скільки підтверджень поспіль потрібно, щоб трек з'явився у виході
                    (відсіює поодинокі хибні спрацювання детектора);
    iou_threshold — мінімальний IoU між детекцією та прогнозом для зіставлення.
    """

    def __init__(self, max_age: int = 15, min_hits: int = 2, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[Track] = []
        self.frame = 0
        self._next_id = 1

    def reset(self) -> None:
        self.tracks, self.frame, self._next_id = [], 0, 1

    def update(self, dets: np.ndarray) -> np.ndarray:
        """Обробляє детекції одного кадру.

        dets (N, 5): x1, y1, x2, y2, score -> підтверджені треки (M, 6): x1, y1, x2, y2, score, track_id.
        Викликати для кожного кадру по порядку, навіть якщо детекцій немає (тоді передати порожній масив).
        """
        self.frame += 1
        dets = np.asarray(dets, np.float64).reshape(-1, 5)
        # 1) Прогноз: де, за фільтром Калмана, зараз має бути кожен відомий пішохід.
        predicted = np.array([t.kf.predict() for t in self.tracks]).reshape(-1, 4)
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        # 2) Зіставлення: угорський алгоритм мінімізує сумарну «вартість» -IoU,
        #    тобто знаходить призначення детекцій трекам з максимальним сумарним IoU.
        matches, unmatched_dets = [], list(range(len(dets)))
        if len(dets) and len(self.tracks):
            iou = box_iou(dets[:, :4], predicted)
            rows, cols = linear_sum_assignment(-iou)
            matched_d = set()
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:  # занадто далекі пари не зіставляємо
                    matches.append((r, c))
                    matched_d.add(r)
            unmatched_dets = [d for d in range(len(dets)) if d not in matched_d]

        # 3) Оновлення зіставлених треків новими вимірами.
        for d, ti in matches:
            t = self.tracks[ti]
            t.kf.update(dets[d, :4])
            t.score = float(dets[d, 4])
            t.hits += 1
            # серія не переривалась, якщо трек мав детекцію й на попередньому кадрі
            t.hit_streak = t.hit_streak + 1 if t.time_since_update == 1 else 1
            t.time_since_update = 0
        # 4) Нові треки для детекцій, які нікому не підійшли (новий пішохід у кадрі).
        for d in unmatched_dets:
            self.tracks.append(Track(self._next_id, KalmanBoxFilter(dets[d, :4]), float(dets[d, 4])))
            self._next_id += 1

        # 5) Видалення треків, які надто довго не підтверджувались (пішохід зник з кадру).
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        # 6) Вихід: лише треки, підтверджені на цьому кадрі та «стабільні» (або перші кадри відео).
        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hit_streak >= self.min_hits or self.frame <= self.min_hits):
                b = t.kf.box
                t.history.append(((b[0] + b[2]) / 2, b[3]))
                t.history = t.history[-30:]  # траєкторія за останні 30 кадрів
                out.append([*b, t.score, t.id])
        return np.array(out, np.float64).reshape(-1, 6)

    def trails(self) -> dict[int, list]:
        """Траєкторії активних треків {id: [(x, y), ...]} для візуалізації."""
        return {t.id: t.history for t in self.tracks if t.time_since_update == 0}
