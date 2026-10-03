"""Detection metrics: Precision/Recall, AP (COCO-style), log-average miss rate (Caltech)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU (Jaccard index) matrix between boxes a (N, 4) and b (M, 4), xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def match_image(dets: np.ndarray, gts: np.ndarray, iou_thr: float,
                min_h: float = 0.0) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy matching by score.

    Returns (scores, is_tp, n_gt) for the image. Ground truth lower than `min_h`
    pixels is an *ignore* region: detections matched to it are neither TP nor FP,
    and unmatched detections lower than `min_h` are dropped (Caltech protocol).
    """
    if len(dets):
        dets = dets[np.argsort(-dets[:, 4])]
    gt_h = gts[:, 3] - gts[:, 1] if len(gts) else np.zeros(0)
    ignore_gt = gt_h < min_h
    iou = box_iou(dets[:, :4], gts) if len(dets) and len(gts) else np.zeros((len(dets), len(gts)))
    used = np.zeros(len(gts), bool)
    scores, tps = [], []
    for i in range(len(dets)):
        best, best_j = iou_thr, -1
        # prefer non-ignored GT
        for j in np.argsort(-iou[i]) if len(gts) else []:
            if iou[i, j] < iou_thr:
                break
            if used[j] or ignore_gt[j]:
                continue
            best, best_j = iou[i, j], j
            break
        if best_j >= 0:
            used[best_j] = True
            scores.append(dets[i, 4]); tps.append(True)
            continue
        if len(gts) and ignore_gt.any() and (iou[i][ignore_gt] >= iou_thr).any():
            continue  # matched to ignore region
        if (dets[i, 3] - dets[i, 1]) < min_h:
            continue
        scores.append(dets[i, 4]); tps.append(False)
    return np.array(scores, np.float32), np.array(tps, bool), int((~ignore_gt).sum())


@dataclass
class EvalResult:
    ap50: float
    ap50_95: float
    precision: float
    recall: float
    f1: float
    best_conf: float
    mr2: float
    pr_curve: tuple[np.ndarray, np.ndarray] = field(repr=False)
    fppi_curve: tuple[np.ndarray, np.ndarray] = field(repr=False)


def _accumulate(per_image, iou_thr: float, min_h: float):
    all_s, all_tp, n_gt = [], [], 0
    for dets, gts in per_image:
        s, tp, n = match_image(dets, gts, iou_thr, min_h)
        all_s.append(s); all_tp.append(tp); n_gt += n
    s = np.concatenate(all_s) if all_s else np.zeros(0)
    tp = np.concatenate(all_tp) if all_tp else np.zeros(0, bool)
    order = np.argsort(-s)
    return s[order], tp[order], n_gt


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """COCO 101-point interpolated AP."""
    if len(recall) == 0:
        return 0.0
    mpre = np.concatenate([[1.0], precision, [0.0]])
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(x, mrec, mpre, right=0) * (x <= recall.max())))


def log_average_miss_rate(fppi: np.ndarray, miss: np.ndarray) -> float:
    """MR^-2 from Dollar et al. (2012): geometric mean of miss rate at 9 FPPI points in [1e-2, 1]."""
    if len(fppi) == 0:
        return 1.0
    refs = np.logspace(-2, 0, 9)
    vals = []
    for r in refs:
        idx = np.where(fppi <= r)[0]
        vals.append(miss[idx[-1]] if len(idx) else 1.0)
    return float(np.exp(np.mean(np.log(np.maximum(vals, 1e-10)))))


def evaluate(per_image: list[tuple[np.ndarray, np.ndarray]], min_h: float = 0.0) -> EvalResult:
    """per_image: list of (dets (N,5) xyxy+score, gts (M,4) xyxy)."""
    n_img = max(1, len(per_image))
    aps = []
    for thr in np.arange(0.5, 0.96, 0.05):
        s, tp, n_gt = _accumulate(per_image, thr, min_h)
        ctp = np.cumsum(tp); cfp = np.cumsum(~tp)
        rec = ctp / max(n_gt, 1)
        prec = ctp / np.maximum(ctp + cfp, 1)
        aps.append(average_precision(rec, prec))
        if abs(thr - 0.5) < 1e-6:
            rec50, prec50, s50, fp50 = rec, prec, s, cfp
    f1 = 2 * prec50 * rec50 / np.maximum(prec50 + rec50, 1e-9)
    bi = int(np.argmax(f1)) if len(f1) else 0
    fppi = fp50 / n_img
    miss = 1 - rec50
    return EvalResult(
        ap50=aps[0], ap50_95=float(np.mean(aps)),
        precision=float(prec50[bi]) if len(f1) else 0.0,
        recall=float(rec50[bi]) if len(f1) else 0.0,
        f1=float(f1[bi]) if len(f1) else 0.0,
        best_conf=float(s50[bi]) if len(f1) else 0.0,
        mr2=log_average_miss_rate(fppi, miss),
        pr_curve=(rec50, prec50), fppi_curve=(fppi, miss),
    )
