from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def annotate_frame(model: YOLO, frame_bgr: np.ndarray, conf: float, iou: float) -> tuple[np.ndarray, dict]:
    r = model.predict(source=frame_bgr, conf=conf, iou=iou, device="mps", verbose=False)[0]

    det_count = 0
    best_conf = 0.0
    if r.boxes is not None and len(r.boxes) > 0:
        det_count = int(len(r.boxes))
        best_conf = float(r.boxes.conf.max().item())

    vis = frame_bgr.copy()
    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names

        for (x1, y1, x2, y2), sc, cid in zip(xyxy, scores, cls_ids):
            label = str(names.get(int(cid), cid))
            if label != "person":
                continue  # показуємо тільки пішоходів
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{label} {sc:.2f}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    summary = {"people_detected": det_count, "best_conf": best_conf}
    return vis, summary


st.set_page_config(page_title="Pedestrian Detector", layout="wide")
st.title("Pedestrian Detection (Baseline vs Fine-tuned)")

models = {
    "Baseline (yolov8n.pt)": "models/yolov8n.pt",
    "Fine-tuned (Caltech)": "models/finetuned_best.pt",
}

with st.sidebar:
    model_choice = st.selectbox("Model", list(models.keys()))
    conf = st.slider("Confidence", 0.0, 1.0, 0.25, 0.01)
    iou = st.slider("IoU", 0.0, 1.0, 0.45, 0.01)
    max_fps = st.slider("UI max FPS", 1, 60, 30, 1)
    infer_every_n = st.slider("Infer every N frames", 1, 10, 1, 1)

model_path = models[model_choice]
if model_path != "models/yolov8n.pt" and not Path(model_path).exists():
    st.error(f"Fine-tuned model not found: {model_path}. Copy best.pt to models/finetuned_best.pt")
    st.stop()


@st.cache_resource
def get_model_cached(path: str) -> YOLO:
    return load_model(path)


model = get_model_cached(model_path)

tab_photo, tab_video = st.tabs(["Photo", "Video"])

with tab_photo:
    img_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if img_file:
        data = np.frombuffer(img_file.read(), dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        vis, summary = annotate_frame(model, frame, conf, iou)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.markdown(
            f"**People detected:** {summary['people_detected']}  \n**Best confidence:** {summary['best_conf']:.2f}")

with tab_video:
    vid_file = st.file_uploader("Upload video (mp4)", type=["mp4", "mov", "m4v", "avi"])
    start = st.button("Start video processing")
    if vid_file and start:
        tmp_path = Path("tmp_input_video")
        tmp_path.mkdir(exist_ok=True)
        in_path = tmp_path / vid_file.name
        in_path.write_bytes(vid_file.read())

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            st.error("Cannot open video.")
            st.stop()

        frame_slot = st.empty()
        stat_slot = st.empty()

        frame_idx = 0
        people_sum = 0
        people_max = 0

        last_ui = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % infer_every_n == 0:
                vis, summary = annotate_frame(model, frame, conf, iou)
            else:
                vis = frame  # без інференсу на цьому кадрі
                summary = {"people_detected": 0, "best_conf": 0.0}

            people = int(summary["people_detected"])
            people_sum += people
            people_max = max(people_max, people)

            now = time.time()
            ui_fps = 1.0 / (now - last_ui) if now > last_ui else 0.0
            last_ui = now

            frame_slot.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), use_container_width=True)
            avg_people = people_sum / max(1, frame_idx + 1)
            stat_slot.markdown(
                f"""
                ### Stats
                - Frame: **{frame_idx}**
                - UI FPS: **{ui_fps:.1f}**
                - People (current): **{people}**
                - People (avg): **{avg_people:.2f}**
                - People (max): **{people_max}**
                """
            )

            frame_idx += 1
            target_dt = 1.0 / float(max_fps)
            spent = time.time() - now
            if spent < target_dt:
                time.sleep(target_dt - spent)

        cap.release()
        st.success("Video processing finished.")
