"""Навчання власної моделі PedNet на Caltech Pedestrian (власний цикл навчання на PyTorch).

Що відбувається:
  * завантажуються кожен 2-й кадр train і кожен 4-й кадр val (сусідні кадри відео майже однакові);
  * оптимізатор AdamW: менший learning rate для попередньо навченого backbone, більший — для нових шарів;
  * розклад швидкості навчання: лінійний warmup, потім косинусне зменшення;
  * після кожної епохи — AP50 на валідації; найкраща модель зберігається в models/pednet_best.pt;
  * історія (втрати та метрики по епохах) пишеться в runs/pednet/history.csv.

Приклади:
    python train_pednet.py --epochs 25
    python train_pednet.py --epochs 1 --limit 200      # швидка перевірка, що все працює
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pedestrian.data import CALTECH_ROOT, PedestrianDataset, collate, list_images
from pedestrian.device import pick_device
from pedestrian.metrics import evaluate
from pedestrian.pednet import PedNet, PedNetConfig, count_parameters, decode, pednet_loss


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-iters", type=int, default=500)
    ap.add_argument("--train-step", type=int, default=2, help="use every N-th train frame")
    ap.add_argument("--val-step", type=int, default=4, help="use every N-th val frame")
    ap.add_argument("--limit", type=int, default=0, help="limit images (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default=pick_device())
    ap.add_argument("--out-dir", default="runs/pednet")
    ap.add_argument("--export-to", default="models/pednet_best.pt")
    ap.add_argument("--resume", default="")
    return ap.parse_args()


@torch.no_grad()
def validate(model: PedNet, loader: DataLoader, device: str) -> dict:
    """Прогін валідаційної вибірки та обчислення метрик власним модулем metrics.py.

    conf=0.01 — низький поріг, щоб побудувати повну криву Precision–Recall.
    """
    model.eval()  # BatchNorm у режимі інференсу
    per_image = []
    for imgs, _, boxes in loader:
        dets = decode(model(imgs.to(device)), stride=model.cfg.stride, conf=0.01)
        per_image += [(d.numpy(), b.numpy()) for d, b in zip(dets, boxes)]
    r = evaluate(per_image)
    r60 = evaluate(per_image, min_h=60)  # GT in this Caltech version: h >= 75 px
    return {"val_ap50": r.ap50, "val_ap50_95": r.ap50_95, "val_precision": r.precision,
            "val_recall": r.recall, "val_mr2": r60.mr2}


def main() -> None:
    a = parse_args()
    # фіксуємо генератори випадкових чисел для відтворюваності
    torch.manual_seed(0)
    np.random.seed(0)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_imgs = list_images(CALTECH_ROOT, "train")[::a.train_step]
    val_imgs = list_images(CALTECH_ROOT, "val")[::a.val_step]
    if a.limit:
        train_imgs, val_imgs = train_imgs[:a.limit], val_imgs[:max(50, a.limit // 4)]
    cfg = PedNetConfig()
    size = (cfg.input_w, cfg.input_h)
    train_ds = PedestrianDataset(train_imgs, size, cfg.stride, augment=True)
    val_ds = PedestrianDataset(val_imgs, size, cfg.stride, augment=False)
    kw = dict(num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, a.batch, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, a.batch, shuffle=False, **kw)

    model = PedNet(cfg).to(a.device)  # backbone ініціалізується вагами ImageNet
    print(f"PedNet: {count_parameters(model) / 1e6:.2f}M parameters | train {len(train_ds)} | val {len(val_ds)} "
          f"| device {a.device}")

    # Дві групи параметрів: backbone вже вміє виділяти ознаки (ImageNet), тому його донавчаємо
    # обережніше (lr * 0.5); FPN і голови навчаються з нуля — повний lr.
    backbone = list(model.backbone.parameters())
    bb_ids = {id(p) for p in backbone}
    rest = [p for p in model.parameters() if id(p) not in bb_ids]
    opt = torch.optim.AdamW([{"params": backbone, "lr": a.lr * 0.5}, {"params": rest, "lr": a.lr}],
                            weight_decay=a.weight_decay)
    base_lrs = [g["lr"] for g in opt.param_groups]
    total_iters = a.epochs * len(train_dl)

    start_epoch, best_ap = 0, -1.0
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        start_epoch, best_ap = ck["epoch"] + 1, ck.get("best_ap", -1.0)

    history_path = out_dir / "history.csv"
    fields = ["epoch", "time_s", "lr", "loss", "hm", "size", "off",
              "val_ap50", "val_ap50_95", "val_precision", "val_recall", "val_mr2"]
    if not a.resume:
        with history_path.open("w", newline="") as f:
            csv.DictWriter(f, fields).writeheader()

    it = start_epoch * len(train_dl)
    for epoch in range(start_epoch, a.epochs):
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "hm": 0.0, "size": 0.0, "off": 0.0}
        for bi, (imgs, t, _) in enumerate(train_dl):
            # Розклад lr: перші warmup_iters ітерацій — лінійне зростання від 0 (стабільний старт,
            # поки голови видають випадкові значення), далі — косинусне зменшення до 1% від початкового.
            f = (it + 1) / a.warmup_iters if it < a.warmup_iters else \
                0.5 * (1 + math.cos(math.pi * (it - a.warmup_iters) / max(1, total_iters - a.warmup_iters)))
            for g, lr0 in zip(opt.param_groups, base_lrs):
                g["lr"] = lr0 * max(f, 0.01)
            imgs = imgs.to(a.device, non_blocking=True)
            t = {k: v.to(a.device, non_blocking=True) for k, v in t.items()}
            # прямий прохід -> функція втрат -> зворотне поширення -> крок оптимізатора
            losses = pednet_loss(model(imgs), t)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)  # захист від «вибуху» градієнтів
            opt.step()
            for k in sums:
                sums[k] += losses[k].item()
            it += 1
            if bi % 50 == 0:
                print(f"ep {epoch + 1}/{a.epochs} it {bi}/{len(train_dl)} loss {float(losses['loss']):.3f} "
                      f"(hm {float(losses['hm']):.3f} size {float(losses['size']):.3f} off {float(losses['off']):.3f}) "
                      f"{(time.time() - t0) / (bi + 1):.2f}s/it", flush=True)

        n = len(train_dl)
        val = validate(model, val_dl, a.device)
        row = {"epoch": epoch + 1, "time_s": round(time.time() - t0, 1), "lr": opt.param_groups[1]["lr"],
               **{k: v / n for k, v in sums.items()}, **val}
        with history_path.open("a", newline="") as f:
            csv.DictWriter(f, fields).writerow(row)
        print(f"== epoch {epoch + 1}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                    for k, v in row.items()), flush=True)

        # last.pt — для продовження навчання (--resume); best.pt — найкраща за val AP50
        model.save(str(out_dir / "last.pt"), epoch=epoch, optimizer=opt.state_dict(), best_ap=best_ap)
        if val["val_ap50"] > best_ap:
            best_ap = val["val_ap50"]
            model.save(str(out_dir / "best.pt"), epoch=epoch, val=val)
            model.save(a.export_to, epoch=epoch, val=val)
            print(f"   new best AP50={best_ap:.4f} -> {a.export_to}", flush=True)


if __name__ == "__main__":
    main()
