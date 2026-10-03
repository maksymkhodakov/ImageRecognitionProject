"""Detect and track pedestrians in a frame sequence (folder of images) or a video.

Output: list of rectangles for every frame (JSON + CSV):
    [{"frame": 0, "file": "...", "boxes": [{"track_id": 1, "x1": .., "y1": .., "x2": .., "y2": .., "conf": ..}]}]

Example:
    python track.py --source dataset/datasets/images/test/caltechpedestriandataset/set06 \
                    --pattern "set06_V016_*" --model pednet --out outputs/set06_V016.json --save-video
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import pandas as pd

from pedestrian.data import IMG_EXTS
from pedestrian.detectors import MODEL_ZOO, Detector, load_detector
from pedestrian.sort_tracker import SortTracker
from pedestrian.viz import draw_tracks


def iter_frames(source: str, pattern: str = "*") -> Iterator[tuple[str, np.ndarray]]:
    """Yield (name, BGR frame) from a directory of images (sorted) or a video file."""
    p = Path(source)
    if p.is_dir():
        for f in sorted(x for x in p.glob(pattern) if x.suffix.lower() in IMG_EXTS):
            yield f.name, cv2.imread(str(f))
        return
    cap = cv2.VideoCapture(str(p))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield f"{p.stem}_{i:06d}", frame
        i += 1
    cap.release()


def run_tracking(detector: Detector, frames, conf: float = 0.3, iou: float = 0.5,
                 tracker: SortTracker | None = None, on_frame=None) -> list[dict]:
    """Core pipeline: detector -> SORT -> per-frame list of boxes with track ids."""
    tracker = tracker or SortTracker()
    results = []
    for idx, (name, frame) in enumerate(frames):
        t0 = time.perf_counter()
        dets = detector.predict(frame, conf, iou)
        tracks = tracker.update(dets)
        ms = (time.perf_counter() - t0) * 1000
        rec = {
            "frame": idx, "file": name, "ms": round(ms, 2),
            "boxes": [{"track_id": int(t[5]), "x1": round(float(t[0]), 1), "y1": round(float(t[1]), 1),
                       "x2": round(float(t[2]), 1), "y2": round(float(t[3]), 1), "conf": round(float(t[4]), 3)}
                      for t in tracks],
        }
        results.append(rec)
        if on_frame is not None:
            on_frame(idx, frame, tracks, tracker, rec)
    return results


def results_to_frame(results: list[dict]) -> pd.DataFrame:
    rows = [{"frame": r["frame"], "file": r["file"], **b} for r in results for b in r["boxes"]]
    return pd.DataFrame(rows, columns=["frame", "file", "track_id", "x1", "y1", "x2", "y2", "conf"])


def track_stats(results: list[dict]) -> dict:
    df = results_to_frame(results)
    n = len(results)
    if df.empty:
        return {"frames": n, "tracks": 0, "avg_track_len": 0.0, "avg_people": 0.0, "max_people": 0,
                "avg_ms": float(np.mean([r["ms"] for r in results])) if results else 0.0}
    lens = df.groupby("track_id").size()
    per_frame = [len(r["boxes"]) for r in results]
    avg_ms = float(np.mean([r["ms"] for r in results]))
    return {"frames": n, "tracks": int(lens.size), "avg_track_len": float(lens.mean()),
            "avg_people": float(np.mean(per_frame)), "max_people": int(np.max(per_frame)),
            "avg_ms": avg_ms, "fps": 1000.0 / max(avg_ms, 1e-6)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="folder with frames or a video file")
    ap.add_argument("--pattern", default="*", help="glob for frames inside the folder")
    ap.add_argument("--model", default="pednet", choices=list(MODEL_ZOO))
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--max-age", type=int, default=15)
    ap.add_argument("--min-hits", type=int, default=2)
    ap.add_argument("--out", default="outputs/tracks.json")
    ap.add_argument("--save-video", action="store_true")
    ap.add_argument("--fps", type=float, default=25.0)
    a = ap.parse_args()

    det = load_detector(a.model)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = None

    def on_frame(idx, frame, tracks, tracker, rec):
        nonlocal writer
        if a.save_video:
            vis = draw_tracks(frame, tracks, tracker.trails())
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(str(out.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
            writer.write(vis)

    results = run_tracking(det, iter_frames(a.source, a.pattern), a.conf, a.iou,
                           SortTracker(a.max_age, a.min_hits), on_frame)
    if writer is not None:
        writer.release()
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    results_to_frame(results).to_csv(out.with_suffix(".csv"), index=False)
    print(json.dumps(track_stats(results), indent=2))
    print(f"Saved: {out} and {out.with_suffix('.csv')}" + (f", {out.with_suffix('.mp4')}" if a.save_video else ""))


if __name__ == "__main__":
    main()
