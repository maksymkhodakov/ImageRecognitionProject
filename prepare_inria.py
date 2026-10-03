"""Download INRIA Person and convert it to YOLO format (used as an independent test set).

Source: Kaggle mirror `jcoral02/inriaperson` (VOC-style XML annotations).
Credentials: ~/.kaggle/access_token (new Kaggle API token) or ~/.kaggle/kaggle.json,
or download the archive manually and pass --archive.

Example:
    python prepare_inria.py
    python prepare_inria.py --archive ~/Downloads/inriaperson.zip
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

KAGGLE_REF = "jcoral02/inriaperson"
RAW_DIR = Path("dataset/raw")
OUT_DIR = Path("dataset/inria")


def kaggle_auth_header() -> str:
    home = Path.home() / ".kaggle"
    if (home / "access_token").exists():
        return "Authorization: Bearer " + (home / "access_token").read_text().strip()
    if (home / "kaggle.json").exists():
        cred = json.loads((home / "kaggle.json").read_text())
        token = base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode()
        return "Authorization: Basic " + token
    raise SystemExit("No Kaggle credentials in ~/.kaggle — download the archive manually and use --archive")


def download(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_REF}"
    print(f"Downloading {KAGGLE_REF} -> {dst}")
    subprocess.run(["curl", "-fL", "-H", kaggle_auth_header(), "-o", str(dst), url], check=True)
    return dst


def parse_voc(xml_bytes: bytes) -> tuple[int, int, list[tuple[float, float, float, float]]]:
    root = ET.fromstring(xml_bytes)
    w = int(root.findtext("size/width"))
    h = int(root.findtext("size/height"))
    boxes = []
    for obj in root.iter("object"):
        if obj.findtext("name", "person").lower() != "person":
            continue
        bb = obj.find("bndbox")
        boxes.append(tuple(float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")))
    return w, h, boxes


def convert(archive: Path, split: str = "Test") -> int:
    img_out = OUT_DIR / "images" / "test"
    lbl_out = OUT_DIR / "labels" / "test"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        for ann in sorted(x for x in names if f"{split}/Annotations/" in x and x.endswith(".xml")):
            w, h, boxes = parse_voc(zf.read(ann))
            stem = Path(ann).stem
            img_name = next((x for x in names if f"{split}/JPEGImages/{stem}." in x), None)
            if img_name is None:
                continue
            with zf.open(img_name) as src, (img_out / Path(img_name).name).open("wb") as dst:
                shutil.copyfileobj(src, dst)
            lines = []
            for x1, y1, x2, y2 in boxes:
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                lines.append(f"0 {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            (lbl_out / f"{Path(img_name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n += 1
    (OUT_DIR.parent / "inria.yaml").write_text(
        f"path: {OUT_DIR.resolve()}\ntrain: images/test\nval: images/test\ntest: images/test\nnc: 1\nnames: ['person']\n"
    )
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", type=Path, default=None)
    a = ap.parse_args()
    archive = a.archive or RAW_DIR / "inriaperson.zip"
    if not archive.exists():
        download(archive)
    n = convert(archive)
    print(f"INRIA test: {n} images -> {OUT_DIR}")


if __name__ == "__main__":
    main()
