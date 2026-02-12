from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

from ultralytics import YOLO


Device = Literal["cpu", "mps", "0"]


@dataclass(frozen=True)
class TrainConfig:
    data_yaml: Path
    base_model: str = "yolov8n.pt"
    epochs: int = 20
    imgsz: int = 640
    batch: int = 8
    device: Device = "mps"
    project: str = "runs"
    name: str = "caltech_ped_finetune"


def main(cfg: TrainConfig) -> None:
    if not cfg.data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {cfg.data_yaml}")

    model = YOLO(cfg.base_model)

    model.train(
        data=str(cfg.data_yaml),
        epochs=cfg.epochs,
        imgsz=cfg.imgsz,
        batch=cfg.batch,
        device=cfg.device,
        project=cfg.project,
        name=cfg.name,
    )

    print("\nTraining finished.")
    print(f"Check weights in: {Path(cfg.project) / 'detect' / cfg.name / 'weights'}")


if __name__ == "__main__":
    cfg = TrainConfig(
        data_yaml=Path("dataset/custom_dataset.yaml"),
        base_model="yolov8n.pt",
        epochs=5,
        imgsz=640,
        batch=8,
        device="mps",
        project="runs",
        name="caltech_ped_test",
    )
    main(cfg)
