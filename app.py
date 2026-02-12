from __future__ import annotations

import time
from pathlib import Path

import tempfile
import os
import cv2
import imageio
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO


def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


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
if "processed_video_bytes" not in st.session_state:
    st.session_state.processed_video_bytes = None

if "processed_csv_bytes" not in st.session_state:
    st.session_state.processed_csv_bytes = None


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
    if "processed_video_bytes" not in st.session_state:
        st.session_state.processed_video_bytes = None
    if "processed_csv_bytes" not in st.session_state:
        st.session_state.processed_csv_bytes = None

    vid_file = st.file_uploader("Upload video (mp4)", type=["mp4", "mov", "m4v", "avi"])
    start = st.button("Start video processing")
    save_annotated = True

    if vid_file and start:
        tmp_path = Path("tmp_input_video")
        tmp_path.mkdir(exist_ok=True)
        in_path = tmp_path / vid_file.name
        in_path.write_bytes(vid_file.read())

        cap = cv2.VideoCapture(str(in_path))
        if not cap.isOpened():
            st.error("Cannot open video.")
            st.stop()

        # read video metadata
        src_fps = cap.get(cv2.CAP_PROP_FPS)
        if not src_fps or src_fps <= 1e-3:
            src_fps = 25.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if width == 0 or height == 0:
            # fallback: read one frame to get shape
            ok, frame0 = cap.read()
            if not ok:
                st.error("Video has no frames.")
                st.stop()
            height, width = frame0.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_path = Path(tmp_out.name)
        tmp_out.close()

        writer = None
        if save_annotated:
            # imageio writer (ffmpeg) - stable on mac
            writer = imageio.get_writer(
                str(out_path),
                fps=float(min(src_fps, max_fps)),
                codec="libx264",
                format="FFMPEG"
            )

        frame_slot = st.empty()
        stat_slot = st.empty()
        table_slot = st.empty()

        # logging per frame
        rows: list[dict] = []

        frame_idx = 0
        people_sum = 0
        people_max = 0

        last_ui = time.time()
        infer_times: list[float] = []

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            t0 = time.time()
            if frame_idx % infer_every_n == 0:
                vis, summary = annotate_frame(model, frame, conf, iou)
            else:
                vis = frame
                summary = {"people_detected": 0, "best_conf": 0.0}
            t1 = time.time()

            infer_ms = (t1 - t0) * 1000.0
            infer_times.append(infer_ms)

            people = int(summary["people_detected"])
            people_sum += people
            people_max = max(people_max, people)

            # UI FPS (rough)
            now = time.time()
            ui_fps = 1.0 / (now - last_ui) if now > last_ui else 0.0
            last_ui = now

            avg_people = people_sum / max(1, frame_idx + 1)
            avg_infer_ms = float(np.mean(infer_times)) if infer_times else 0.0

            # show frame
            frame_slot.image(bgr_to_rgb(vis), use_container_width=True)

            stat_slot.markdown(
                f"""
                ### Stats
                - Frame: **{frame_idx}**
                - UI FPS: **{ui_fps:.1f}**
                - Inference avg (ms): **{avg_infer_ms:.1f}**
                - People (current): **{people}**
                - People (avg): **{avg_people:.2f}**
                - People (max): **{people_max}**
                """
            )

            # write annotated video
            if writer is not None:
                writer.append_data(bgr_to_rgb(vis))

            # log row
            rows.append(
                {
                    "frame": frame_idx,
                    "people": people,
                    "best_conf": float(summary["best_conf"]),
                    "infer_ms": float(infer_ms),
                    "ui_fps": float(ui_fps),
                }
            )

            frame_idx += 1

            # throttle UI
            target_dt = 1.0 / float(max_fps)
            spent = time.time() - now
            if spent < target_dt:
                time.sleep(target_dt - spent)

            # light live table update every ~30 frames
            if frame_idx % 30 == 0:
                df_live = pd.DataFrame(rows[-300:])  # last 300 frames
                table_slot.dataframe(df_live, use_container_width=True)

        cap.release()
        if writer is not None:
            writer.close()

        # final logs
        df = pd.DataFrame(rows)

        st.success("Video processing finished.")

        st.subheader("Summary")
        st.write(
            {
                "frames": int(df.shape[0]),
                "avg_people": float(df["people"].mean()) if not df.empty else 0.0,
                "max_people": int(df["people"].max()) if not df.empty else 0,
                "avg_infer_ms": float(df["infer_ms"].mean()) if not df.empty else 0.0,
            }
        )

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.session_state.processed_csv_bytes = csv_bytes

        st.subheader("Per-frame log (CSV)")
        st.dataframe(df.head(50), use_container_width=True)

        if save_annotated and out_path.exists():
            video_bytes = out_path.read_bytes()
            st.session_state.processed_video_bytes = video_bytes

            # clean temp file
            try:
                os.remove(out_path)
            except Exception:
                pass

    if st.session_state.processed_video_bytes is not None:
        st.subheader("Annotated video")
        st.video(st.session_state.processed_video_bytes)
        st.download_button(
            "Download annotated video",
            data=st.session_state.processed_video_bytes,
            file_name="annotated_video.mp4",
            mime="video/mp4",
        )

    if st.session_state.processed_csv_bytes is not None:
        st.download_button(
            "Download CSV log",
            data=st.session_state.processed_csv_bytes,
            file_name="video_stats.csv",
            mime="text/csv",
        )
