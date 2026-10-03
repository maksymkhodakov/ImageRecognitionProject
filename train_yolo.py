"""Донавчання YOLOv8n на Caltech Pedestrian — модель для порівняння з власною PedNet.

Кадри Caltech — послідовні кадри відео, сусідні майже однакові. Щоб навчання на Apple M1
тривало розумний час, використовується кожен `--train-step`-й кадр для навчання
та кожен `--val-step`-й — для валідації (списки файлів пишуться в dataset/lists/).
Навчання виконує бібліотека Ultralytics; ми лише готуємо дані й параметри.

Приклади:
    python train_yolo.py --epochs 12 --name caltech_v8n_e20
    python train_yolo.py --resume runs/detect/runs/yolo/caltech_v8n_e20/weights/last.pt
"""
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from pedestrian.data import CALTECH_ROOT, list_images
from pedestrian.device import pick_device


@dataclass(frozen=True)
class TrainConfig:
    """Параметри навчання (значення за замовчуванням — ті, з якими навчено модель у звіті)."""
    base_model: str = "models/yolov8n.pt"
    epochs: int = 12
    imgsz: int = 640
    batch: int = 16
    train_step: int = 2
    val_step: int = 4
    patience: int = 8
    device: str = pick_device()
    workers: int = 6
    project: str = "runs/yolo"
    name: str = "caltech_v8n"
    export_to: str = "models/yolo_caltech_v8n.pt"


def write_subset_yaml(cfg: TrainConfig) -> Path:
    """Пише списки зображень (кожен N-й кадр) і data.yaml для Ultralytics, що на них посилається."""
    out_dir = Path("dataset/lists")
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, step in (("train", cfg.train_step), ("val", cfg.val_step)):
        imgs = list_images(CALTECH_ROOT, split)[::step]
        p = out_dir / f"caltech_{split}_s{step}.txt"
        p.write_text("\n".join(str(i.resolve()) for i in imgs) + "\n")
        paths[split] = p.resolve()
        print(f"{split}: {len(imgs)} images -> {p}")

    yaml_path = out_dir / "caltech_subset.yaml"
    yaml_path.write_text(
        f"train: {paths['train']}\n"
        f"val: {paths['val']}\n"
        "nc: 1\n"
        "names: ['person']\n"
    )
    return yaml_path


def main(cfg: TrainConfig, resume: str = "") -> None:
    if resume:  # продовження перерваного навчання з його last.pt
        model = YOLO(resume)
        model.train(resume=True)
        export_best(model, cfg)
        return
    data_yaml = write_subset_yaml(cfg)
    model = YOLO(cfg.base_model)
    model.train(
        data=str(data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        workers=cfg.workers,
        patience=cfg.patience,
        cos_lr=True,        # косинусний розклад learning rate
        single_cls=True,    # один клас — person
        project=cfg.project,
        name=cfg.name,
        exist_ok=True,
        seed=0,
        plots=True,
    )
    export_best(model, cfg)


def export_best(model: YOLO, cfg: TrainConfig) -> None:
    """Копіює найкращі ваги (за val mAP) у models/, звідки їх бере решта проєкту."""
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if best.exists():
        shutil.copy(best, cfg.export_to)
        print(f"\nTraining finished. Best weights copied to {cfg.export_to}")


def parse_args() -> tuple[str, TrainConfig]:
    d = TrainConfig()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-model", default=d.base_model)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--imgsz", type=int, default=d.imgsz)
    ap.add_argument("--batch", type=int, default=d.batch)
    ap.add_argument("--train-step", type=int, default=d.train_step)
    ap.add_argument("--val-step", type=int, default=d.val_step)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--device", default=d.device)
    ap.add_argument("--workers", type=int, default=d.workers)
    ap.add_argument("--name", default=d.name)
    ap.add_argument("--export-to", default=d.export_to)
    ap.add_argument("--resume", default="", help="path to last.pt of an interrupted run")
    a = ap.parse_args()
    return a.resume, TrainConfig(
        base_model=a.base_model, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
        train_step=a.train_step, val_step=a.val_step, patience=a.patience,
        device=a.device, workers=a.workers, name=a.name, export_to=a.export_to,
    )


if __name__ == "__main__":
    resume_from, config = parse_args()
    main(config, resume_from)
