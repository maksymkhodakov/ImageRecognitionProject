"""Оцінка всіх моделей на Caltech test та INRIA test за єдиним протоколом.

Для кожної моделі та кожного датасету: детекції з низьким порогом (0.001) на всіх кадрах ->
власні метрики (pedestrian/metrics.py) -> таблиці та графіки для звіту і веб-інтерфейсу.

Результати (папка reports/):
    metrics.csv           P, R, F1, AP50, AP50-95, MR^-2, швидкість, розмір — для кожної пари модель x датасет
    curves.npz            криві PR та MR–FPPI (для інтерактивних графіків у веб-інтерфейсі)
    curves_<ds>.png, speed.png, training_curves.png — графіки для звіту
    tracking.csv          статистика SORT на найдовших послідовностях Caltech test
    samples/              приклади кадрів з детекціями всіх моделей поруч

Приклади:
    python evaluate.py                 # повна оцінка
    python evaluate.py --step 10       # швидко: кожен 10-й кадр Caltech
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pedestrian.data import CALTECH_ROOT, INRIA_ROOT, list_images, read_boxes
from pedestrian.detectors import MODEL_ZOO, available_models, load_detector
from pedestrian.metrics import evaluate
from pedestrian.sort_tracker import SortTracker
from pedestrian.viz import draw_detections
from track import iter_frames, run_tracking, track_stats

REPORTS = Path("reports")
COLORS = {"pednet": "#2a78d6", "yolo_caltech": "#e0702b", "yolo_old": "#8a63c9", "yolo_coco": "#7a7a7a"}
SHORT = {"pednet": "PedNet (own)", "yolo_caltech": "YOLOv8n Caltech-12ep",
         "yolo_old": "YOLOv8n Caltech-5ep", "yolo_coco": "YOLOv8n COCO"}


def collect(det, images: list[Path], warmup: int = 5):
    """Запускає детектор на всіх зображеннях; повертає пари (детекції, істина) і середній час на кадр.

    Перші `warmup` кадрів не враховуються в часі: перший прогін на GPU містить ініціалізацію.
    """
    per_image, times = [], []
    for i, p in enumerate(images):
        img = cv2.imread(str(p))
        boxes, ms = det.timed_predict(img, conf=0.001, iou=0.5)
        if i >= warmup:
            times.append(ms)
        h, w = img.shape[:2]
        per_image.append((boxes, read_boxes(p, w, h)))
    return per_image, float(np.mean(times)) if times else 0.0


def plot_curves(curves: dict, ds: str) -> None:
    """Криві Precision–Recall та Miss rate–FPPI (логарифмічні осі, як у бенчмарку Caltech)."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, c in curves.items():
        if c["dataset"] != ds:
            continue
        lbl = SHORT[c["model"]]
        ax[0].plot(c["rec"], c["prec"], color=COLORS[c["model"]], lw=2, label=f"{lbl} (AP50={c['ap50']:.3f})")
        ax[1].loglog(np.maximum(c["fppi"], 1e-3), np.maximum(c["miss"], 1e-3), color=COLORS[c["model"]], lw=2,
                     label=f"{lbl} (MR⁻²={c['mr2'] * 100:.1f}%)")
    ax[0].set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.02), title=f"Precision-Recall, {ds}")
    ax[1].set(xlabel="False positives per image", ylabel="Miss rate", xlim=(1e-2, 10), ylim=(0.03, 1),
              title=f"Miss rate vs FPPI, {ds}")
    for a in ax:
        a.grid(alpha=0.3, which="both")
        a.legend(fontsize=8, loc="lower left" if a is ax[0] else "lower left")
    fig.tight_layout()
    fig.savefig(REPORTS / f"curves_{ds.lower()}.png", dpi=150)
    plt.close(fig)


def plot_speed(df: pd.DataFrame) -> None:
    """Горизонтальна діаграма часу обробки кадру для кожної моделі."""
    d = df[df.dataset == "Caltech"].sort_values("ms_per_frame")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh([SHORT[m] for m in d.model], d.ms_per_frame, color=[COLORS[m] for m in d.model])
    for i, v in enumerate(d.ms_per_frame):
        ax.text(v + 0.5, i, f"{v:.1f} ms  ({1000 / v:.0f} FPS)", va="center", fontsize=9)
    ax.set(xlabel="ms / frame (Apple M1, MPS, 640x480)", title="Inference speed")
    ax.set_xlim(0, d.ms_per_frame.max() * 1.45)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "speed.png", dpi=150)
    plt.close(fig)


def plot_training() -> None:
    """Криві навчання PedNet (history.csv) та YOLO (results.csv від Ultralytics) на одному рисунку."""
    ped = Path("runs/pednet/history.csv")
    yolo = sorted(Path("runs").rglob("caltech_v8n*/results.csv"), key=lambda f: f.stat().st_mtime)[-1:]
    if not ped.exists() and not yolo:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if ped.exists():
        h = pd.read_csv(ped)
        ax[0].plot(h.epoch, h.loss, color=COLORS["pednet"], lw=2, label="PedNet: train loss")
        ax[1].plot(h.epoch, h.val_ap50, color=COLORS["pednet"], lw=2, label="PedNet: val AP50")
    if yolo:
        y = pd.read_csv(yolo[0])
        y.columns = [c.strip() for c in y.columns]
        tl = y["train/box_loss"] + y["train/cls_loss"] + y["train/dfl_loss"]
        ax[0].plot(y.epoch, tl, color=COLORS["yolo_caltech"], lw=2, label="YOLOv8n: train loss (box+cls+dfl)")
        ax[1].plot(y.epoch, y["metrics/mAP50(B)"], color=COLORS["yolo_caltech"], lw=2, label="YOLOv8n: val mAP50")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Training loss")
    ax[1].set(xlabel="epoch", ylabel="AP50", title="Validation AP50 (Caltech val)")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(REPORTS / "training_curves.png", dpi=150)
    plt.close(fig)


def save_samples(models: list[str], detectors: dict, n: int = 4) -> None:
    """Зберігає сітки «той самий кадр — різні моделі» для візуального порівняння."""
    out = REPORTS / "samples"
    out.mkdir(parents=True, exist_ok=True)
    imgs = list_images(CALTECH_ROOT, "test")
    picks = [imgs[i] for i in np.linspace(200, len(imgs) - 200, n).astype(int)]
    inria = list_images(INRIA_ROOT, "test")
    if inria:
        picks += [inria[5], inria[40]]
    for p in picks:
        frame = cv2.imread(str(p))
        tiles = []
        for m in models:
            vis = draw_detections(frame, detectors[m].predict(frame, 0.3, 0.5))
            cv2.putText(vis, SHORT[m], (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(cv2.resize(vis, (640, int(640 * frame.shape[0] / frame.shape[1]))))
        grid = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
        if len(grid) > 1 and grid[-1].shape != grid[0].shape:
            grid[-1] = np.hstack([grid[-1], np.zeros_like(tiles[0])])
        cv2.imwrite(str(out / f"{p.stem}.jpg"), np.vstack(grid), [cv2.IMWRITE_JPEG_QUALITY, 88])


def longest_sequences(n: int = 5) -> list[str]:
    """n найдовших тестових послідовностей (кадри одного відео), упорядковані за назвою."""
    seqs = pd.Series([p.stem.rsplit("_", 1)[0] for p in list_images(CALTECH_ROOT, "test")]).value_counts()
    return sorted(seqs.index[:n])


def tracking_eval(models: list[str], detectors: dict) -> pd.DataFrame:
    """Повна система (детектор + SORT) на найдовших послідовностях; ID у розмітці немає,
    тому рахуємо статистику треків (кількість, довжина, швидкість), а не MOTA."""
    root = CALTECH_ROOT / "images" / "test" / "caltechpedestriandataset"
    rows = []
    for m, seq in itertools.product(models, longest_sequences()):
        folder = root / seq.split("_")[0]
        res = run_tracking(detectors[m], iter_frames(str(folder), f"{seq}_*"), conf=0.3, tracker=SortTracker())
        if not res:
            continue
        rows.append({"model": m, "sequence": seq, **track_stats(res)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step", type=int, default=1, help="use every N-th Caltech test frame")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--skip-tracking", action="store_true")
    a = ap.parse_args()

    REPORTS.mkdir(exist_ok=True)
    models = a.models or [m for m in MODEL_ZOO if m in available_models()]
    datasets = {"Caltech": list_images(CALTECH_ROOT, "test")[::a.step], "INRIA": list_images(INRIA_ROOT, "test")}
    detectors = {m: load_detector(m) for m in models}

    rows, curves = [], {}
    for m in models:
        det = detectors[m]
        for ds, images in datasets.items():
            if not images:
                continue
            per_image, ms = collect(det, images)
            # "all" — усі детекції; "h60" — протокол Caltech: розмічено лише пішоходів >= 75 px,
            # тому незіставлені детекції нижчі за 60 px (75 / 1.25) не рахуються хибними
            for subset, min_h in (("all", 0.0), ("h60", 60.0)):
                if ds == "INRIA" and subset == "h60":
                    continue
                r = evaluate(per_image, min_h=min_h)
                rows.append({"model": m, "title": det.info.title, "origin": det.info.origin, "dataset": ds,
                             "subset": subset, "images": len(images), "precision": r.precision,
                             "recall": r.recall, "f1": r.f1, "ap50": r.ap50, "ap50_95": r.ap50_95, "mr2": r.mr2,
                             "best_conf": r.best_conf, "ms_per_frame": ms, "fps": 1000 / max(ms, 1e-6),
                             "params_m": det.n_params() / 1e6, "size_mb": det.size_mb()})
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
                if subset == "all":
                    curves[f"{m}|{ds}"] = {"model": m, "dataset": ds, "rec": r.pr_curve[0], "prec": r.pr_curve[1],
                                           "fppi": r.fppi_curve[0], "miss": r.fppi_curve[1],
                                           "ap50": r.ap50, "mr2": r.mr2}

    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "metrics.csv", index=False)
    # криві проріджуються до ~400 точок, щоб файл був невеликим
    flat = {}
    for k, c in curves.items():
        idx = np.unique(np.linspace(0, max(len(c["rec"]) - 1, 0), 400).astype(int)) if len(c["rec"]) else []
        for f in ("rec", "prec", "fppi", "miss"):
            flat[f"{k}|{f}"] = np.asarray(c[f])[idx] if len(c[f]) else np.zeros(0)
    np.savez_compressed(REPORTS / "curves.npz", **flat)
    for ds in datasets:
        plot_curves(curves, ds)
    plot_speed(df)
    plot_training()
    save_samples(models, detectors)
    if not a.skip_tracking:
        tr = tracking_eval([m for m in ("pednet", "yolo_caltech") if m in models], detectors)
        tr.to_csv(REPORTS / "tracking.csv", index=False)
        print(tr.to_string())
    print(f"\nSaved to {REPORTS.resolve()}")


if __name__ == "__main__":
    main()
