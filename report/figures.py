"""Static figures for the report: pipeline scheme, PedNet architecture, dataset examples, heatmap demo."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pedestrian.data import CALTECH_ROOT, INRIA_ROOT, PedestrianDataset, list_images, read_boxes  # noqa: E402

OUT = Path("report/figures")
BLUE, ORANGE, GREY, GREEN = "#2a78d6", "#e0702b", "#6b6b6b", "#2e9e5b"


def box(ax, x, y, w, h, text, color, sub=None, fs=10):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=color + "22", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h / 2 + (0.09 if sub else 0), text, ha="center", va="center", fontsize=fs,
            weight="bold", color="#222")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.16, sub, ha="center", va="center", fontsize=fs - 2, color="#444")


def arrow(ax, x1, y1, x2, y2, color="#444"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color=color))


def pipeline():
    fig, ax = plt.subplots(figsize=(11, 1.6))
    ax.set_xlim(0, 11); ax.set_ylim(0.6, 1.95); ax.axis("off")
    steps = [("Input frames", "dashcam video,\n640x480", GREY), ("Preprocessing", "letterbox,\nnormalisation", GREY),
             ("Detector", "PedNet (own) /\nYOLOv8n", BLUE), ("Post-processing", "peak extraction,\nNMS", BLUE),
             ("Tracker SORT", "Kalman filter +\nHungarian (IoU)", ORANGE), ("Output", "boxes + track ID\nper frame (JSON)", GREEN)]
    w, gap = 1.55, 0.27
    for i, (t, s, c) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        box(ax, x, 0.75, w, 1.1, t, c, s, fs=9.5)
        if i:
            arrow(ax, x - gap + 0.02, 1.3, x - 0.02, 1.3)
    fig.tight_layout()
    fig.savefig(OUT / "pipeline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.4); ax.axis("off")
    box(ax, 0.1, 2.2, 1.3, 1.0, "Input", GREY, "3x480x640")
    stages = [("C2", "24 x 120x160", "s=4"), ("C3", "40 x 60x80", "s=8"), ("C4", "112 x 30x40", "s=16"),
              ("C5", "960 x 15x20", "s=32")]
    ax.text(2.55, 5.15, "Backbone: MobileNetV3-Large\n(ImageNet weights)", ha="center", va="top", fontsize=9.5,
            color=GREY, weight="bold")
    ax.text(5.35, 5.15, "Neck: own FPN\n(depthwise-separable)", ha="center", va="top", fontsize=9.5, color=BLUE,
            weight="bold")
    ax.text(8.9, 5.15, "Heads (own), stride 4", ha="center", va="top", fontsize=9.5, color=ORANGE, weight="bold")
    ys = [0.35, 1.45, 2.55, 3.65]
    for (n, sh, st), y in zip(stages, ys):
        box(ax, 1.9, y, 1.3, 0.85, n, GREY, f"{sh}\n{st}", fs=9)
    arrow(ax, 1.4, 2.7, 1.88, 0.8)
    for i in range(3):
        arrow(ax, 2.55, ys[i] + 0.86, 2.55, ys[i + 1] - 0.01)
    for (n, _, _), y in zip(stages, ys):
        p = "P" + n[1]
        box(ax, 4.7, y, 1.3, 0.85, p, BLUE, "96 ch", fs=9)
        arrow(ax, 3.22, y + 0.42, 4.68, y + 0.42, BLUE)
        ax.text(3.95, y + 0.5, "1x1 conv", fontsize=7.5, ha="center", color=BLUE)
    for i in range(3, 0, -1):
        arrow(ax, 6.05, ys[i] + 0.3, 6.05, ys[i - 1] + 0.6, BLUE)
    ax.text(6.15, 2.2, "upsample x2\n+ add\n+ DW block", fontsize=7.5, color=BLUE)
    heads = [("Heatmap", "1 x 120x160\nfocal loss"), ("Size", "2 x 120x160\nlog w, log h, L1"),
             ("Offset", "2 x 120x160\nsub-pixel, L1")]
    for (n, s), y in zip(heads, (3.3, 1.9, 0.5)):
        box(ax, 8.0, y, 1.8, 0.95, n, ORANGE, s, fs=9)
        arrow(ax, 6.02, 0.78, 7.98, y + 0.47, ORANGE)
    ax.text(10.4, 2.3, "peaks\n3x3 max-pool\n+ top-K\n+ NMS\n->\nboxes", ha="center", va="center", fontsize=8.5,
            color=GREEN, weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "pednet_architecture.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def dataset_examples():
    cal = list_images(CALTECH_ROOT, "train")
    inr = list_images(INRIA_ROOT, "test")
    picks = [cal[300], cal[5200], cal[9800]] + ([inr[10], inr[60], inr[120]] if inr else [])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.2))
    for ax, p in zip(axes.ravel(), picks):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        for x1, y1, x2, y2 in read_boxes(p, w, h):
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (60, 220, 60), 2)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(("Caltech: " if "caltech" in str(p) else "INRIA: ") + p.stem, fontsize=9)
        ax.axis("off")
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "dataset_examples.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def heatmap_demo():
    imgs = list_images(CALTECH_ROOT, "train")
    ds = PedestrianDataset(imgs, augment=False)
    i = next(k for k in range(2000, 4000, 7) if len(ds[k][2]) >= 3)
    x, t, boxes = ds[i]
    img = cv2.imread(str(imgs[i]))
    for x1, y1, x2, y2 in boxes.numpy():
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (60, 220, 60), 2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)); ax[0].set_title("Frame with ground truth"); ax[0].axis("off")
    hm = cv2.resize(t["hm"][0].numpy(), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap="gray", alpha=0.55)
    ax[1].imshow(np.ma.masked_less(hm, 0.02), cmap="autumn", alpha=0.95)
    ax[1].set_title("Target heatmap of centres (stride 4, 120x160), overlay")
    ax[1].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "heatmap_target.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline(); architecture(); dataset_examples(); heatmap_demo()
    print("figures ->", OUT)
