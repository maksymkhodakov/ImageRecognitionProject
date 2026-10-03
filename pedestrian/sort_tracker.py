"""SORT multi-object tracker (Bewley et al., 2016) implemented from scratch.

Each track holds a constant-velocity Kalman filter over the state
    [cx, cy, s, r, vcx, vcy, vs]   (s = area, r = aspect ratio w/h, constant)
Detections are associated to predicted tracks by IoU with the Hungarian algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from pedestrian.metrics import box_iou


def xyxy_to_z(b: np.ndarray) -> np.ndarray:
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([b[0] + w / 2, b[1] + h / 2, w * h, w / max(h, 1e-6)], np.float64)


def x_to_xyxy(x: np.ndarray) -> np.ndarray:
    s, r = max(x[2], 1e-6), max(x[3], 1e-6)
    w = np.sqrt(s * r)
    h = s / w
    return np.array([x[0] - w / 2, x[1] - h / 2, x[0] + w / 2, x[1] + h / 2])


class KalmanBoxFilter:
    """Linear Kalman filter with constant velocity for cx, cy and area."""

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
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return x_to_xyxy(self.x)

    def update(self, box: np.ndarray) -> None:
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
    id: int
    kf: KalmanBoxFilter
    score: float
    hits: int = 1
    hit_streak: int = 1
    age: int = 0
    time_since_update: int = 0
    history: list = field(default_factory=list)  # centre points for drawing trails


class SortTracker:
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
        """dets (N, 5) xyxy+score -> confirmed tracks (M, 6) xyxy+score+track_id."""
        self.frame += 1
        dets = np.asarray(dets, np.float64).reshape(-1, 5)
        predicted = np.array([t.kf.predict() for t in self.tracks]).reshape(-1, 4)
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1

        matches, unmatched_dets = [], list(range(len(dets)))
        if len(dets) and len(self.tracks):
            iou = box_iou(dets[:, :4], predicted)
            rows, cols = linear_sum_assignment(-iou)
            matched_d = set()
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    matches.append((r, c))
                    matched_d.add(r)
            unmatched_dets = [d for d in range(len(dets)) if d not in matched_d]

        for d, ti in matches:
            t = self.tracks[ti]
            t.kf.update(dets[d, :4])
            t.score = float(dets[d, 4])
            t.hits += 1
            t.hit_streak = t.hit_streak + 1 if t.time_since_update == 1 else 1
            t.time_since_update = 0
        for d in unmatched_dets:
            self.tracks.append(Track(self._next_id, KalmanBoxFilter(dets[d, :4]), float(dets[d, 4])))
            self._next_id += 1

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        out = []
        for t in self.tracks:
            if t.time_since_update == 0 and (t.hit_streak >= self.min_hits or self.frame <= self.min_hits):
                b = t.kf.box
                t.history.append(((b[0] + b[2]) / 2, b[3]))
                t.history = t.history[-30:]
                out.append([*b, t.score, t.id])
        return np.array(out, np.float64).reshape(-1, 6)

    def trails(self) -> dict[int, list]:
        return {t.id: t.history for t in self.tracks if t.time_since_update == 0}
