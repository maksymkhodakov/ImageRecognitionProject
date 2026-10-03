"""PedVision — веб-платформа для виявлення та відстеження пішоходів (Streamlit).

Вкладки: Огляд · Фото · Відео й трекінг · Послідовність кадрів · Порівняння моделей · Навчання і метрики.
Уся логіка детекції/трекінгу береться з пакета pedestrian і track.py — тут лише інтерфейс.

Запуск:
    python -m streamlit run app.py
"""
from __future__ import annotations

import io
import json
import tempfile
import time
import zipfile
from pathlib import Path

import altair as alt
import cv2
import imageio
import numpy as np
import pandas as pd
import streamlit as st

from pedestrian.data import CALTECH_ROOT, IMG_EXTS, INRIA_ROOT, list_images
from pedestrian.detectors import MODEL_ZOO, ORIGIN_LABELS, available_models, load_detector
from pedestrian.device import pick_device
from pedestrian.sort_tracker import SortTracker
from pedestrian.viz import draw_detections, draw_tracks
from track import results_to_frame, run_tracking, track_stats

REPORTS = Path("reports")
CALTECH_TEST = CALTECH_ROOT / "images" / "test" / "caltechpedestriandataset"
MAX_KEEP_FRAMES = 600  # скільки анотованих кадрів тримати в пам'яті для переглядача кадрів

st.set_page_config(page_title="PedVision — виявлення пішоходів", page_icon=":material/directions_walk:",
                   layout="wide", initial_sidebar_state="expanded")

# Власні стилі: банер-заголовок, бейджі типу моделі, схема конвеєра, картки метрик.
# Кольори напівпрозорі, тому інтерфейс коректно виглядає і в світлій, і в темній темі.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}
      .hero {padding: 1.4rem 1.6rem; border-radius: 1rem; margin-bottom: 1rem;
             background: linear-gradient(120deg, rgba(42,120,214,.16), rgba(42,120,214,.03) 60%);
             border: 1px solid rgba(42,120,214,.25);}
      .hero h1 {font-size: 1.9rem; margin: 0 0 .3rem 0; padding: 0;}
      .hero p {margin: 0; opacity: .8; font-size: 1.02rem;}
      .badge {display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .78rem;
              font-weight: 600; margin-right: .35rem; border: 1px solid transparent;}
      .badge-own {background: rgba(42,120,214,.15); color: #2a78d6; border-color: rgba(42,120,214,.4);}
      .badge-fine-tuned {background: rgba(224,112,43,.15); color: #d0661f; border-color: rgba(224,112,43,.4);}
      .badge-pretrained {background: rgba(128,128,128,.15); color: #8a8a8a; border-color: rgba(128,128,128,.4);}
      .flow {display: flex; flex-wrap: wrap; align-items: center; gap: .5rem; margin: .4rem 0 1rem;}
      .flow .step {padding: .55rem .9rem; border-radius: .7rem; border: 1px solid rgba(128,128,128,.3);
                   background: rgba(128,128,128,.07); font-size: .92rem;}
      .flow .step b {display: block; font-size: .78rem; opacity: .65; font-weight: 500;}
      .flow .arrow {opacity: .5; font-size: 1.2rem;}
      .muted {opacity: .7; font-size: .9rem;}
      div[data-testid="stMetric"] {background: rgba(128,128,128,.06);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- допоміжні функції
# cache_resource: модель завантажується один раз і спільно використовується між перезапусками скрипта
# (Streamlit перезапускає весь файл після кожної дії користувача).
@st.cache_resource(show_spinner="Завантаження моделі…")
def get_detector(key: str):
    return load_detector(key)


@st.cache_data
def load_metrics() -> pd.DataFrame | None:
    p = REPORTS / "metrics.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_curves() -> dict | None:
    p = REPORTS / "curves.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return {k: z[k] for k in z.files}


@st.cache_data
def caltech_sequences() -> pd.DataFrame:
    rows = []
    for f in sorted(CALTECH_TEST.rglob("*.png")):
        rows.append(f.stem.rsplit("_", 1)[0])
    s = pd.Series(rows).value_counts().sort_index()
    return pd.DataFrame({"sequence": s.index, "frames": s.values})


def rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def decode_upload(file) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(file.getvalue(), np.uint8), cv2.IMREAD_COLOR)


def badge(origin: str) -> str:
    return f'<span class="badge badge-{origin}">{ORIGIN_LABELS[origin]}</span>'


def boxes_table(boxes: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(np.round(boxes[:, :5], 2), columns=["x1", "y1", "x2", "y2", "conf"]) \
        if len(boxes) else pd.DataFrame(columns=["x1", "y1", "x2", "y2", "conf"])


def random_sample(dataset: str) -> Path | None:
    imgs = list_images(CALTECH_ROOT, "test") if dataset == "Caltech" else list_images(INRIA_ROOT, "test")
    return imgs[np.random.randint(len(imgs))] if imgs else None


MODEL_SHORT = {"pednet": "PedNet", "yolo_caltech": "YOLOv8n Caltech-12", "yolo_old": "YOLOv8n Caltech-5",
               "yolo_coco": "YOLOv8n COCO"}
MODEL_COLORS = {"pednet": "#2a78d6", "yolo_caltech": "#e0702b", "yolo_old": "#8a63c9", "yolo_coco": "#8a8a8a"}


# ---------------------------------------------------------------- sidebar
models = available_models()
if not models:
    st.error("Не знайдено жодних ваг моделей у папці `models/`. Запустіть `train_pednet.py` або `train_yolo.py`.")
    st.stop()

with st.sidebar:
    st.markdown("### :material/directions_walk: PedVision")
    st.caption("Виявлення та відстеження пішоходів на відео з автомобільної камери")
    model_key = st.selectbox("Модель", models, format_func=lambda k: MODEL_ZOO[k].title)
    info = MODEL_ZOO[model_key]
    st.markdown(badge(info.origin), unsafe_allow_html=True)
    st.caption(info.description)

    st.divider()
    conf = st.slider("Поріг впевненості", 0.05, 0.95, 0.30, 0.05,
                     help="Мінімальна впевненість моделі, щоб вважати прямокутник пішоходом.")
    iou = st.slider("Поріг NMS (IoU)", 0.1, 0.9, 0.5, 0.05,
                    help="Прямокутники, що перекриваються сильніше, зливаються в один.")
    with st.expander("Параметри трекера SORT"):
        max_age = st.slider("max_age (кадрів без детекції)", 1, 60, 15,
                            help="Скільки кадрів трек живе без підтвердження детекцією.")
        min_hits = st.slider("min_hits (підтвердження)", 1, 10, 2,
                             help="Скільки послідовних детекцій потрібно, щоб трек з'явився.")
    st.divider()
    st.caption(f"Пристрій обчислень: **{pick_device().upper()}**")

detector = get_detector(model_key)

st.markdown(
    f"""<div class="hero"><h1>Виявлення пішоходів</h1>
    <p>Локалізація та відстеження пішоходів на кадрах з камери на лобовому склі. Активна модель:
    <b>{info.title}</b> {badge(info.origin)}</p></div>""",
    unsafe_allow_html=True,
)

tabs = st.tabs([":material/home: Огляд", ":material/image: Фото", ":material/movie: Відео й трекінг",
                ":material/burst_mode: Послідовність кадрів", ":material/compare: Порівняння моделей",
                ":material/monitoring: Навчання і метрики"])

# ---------------------------------------------------------------- overview
with tabs[0]:
    st.markdown("#### Як це працює")
    st.markdown(
        """<div class="flow">
        <div class="step"><b>Вхід</b>Послідовність кадрів / відео</div><div class="arrow">→</div>
        <div class="step"><b>Детектор</b>PedNet або YOLOv8</div><div class="arrow">→</div>
        <div class="step"><b>Трекер</b>SORT: Калман + угорський алгоритм</div><div class="arrow">→</div>
        <div class="step"><b>Вихід</b>Список прямокутників з ID для кожного кадру</div>
        </div>""",
        unsafe_allow_html=True,
    )
    df_m = load_metrics()
    if df_m is not None:
        st.markdown("#### Ключові результати (Caltech test)")
        cal = df_m[(df_m.dataset == "Caltech") & (df_m.subset == "all")].set_index("model")
        cols = st.columns(len(cal))
        for col, (k, r) in zip(cols, cal.iterrows()):
            with col:
                st.markdown(f"**{MODEL_SHORT.get(k, k)}** {badge(r.origin)}", unsafe_allow_html=True)
                st.metric("AP50", f"{r.ap50:.3f}", border=True)
                st.metric("Швидкість", f"{r.fps:.0f} FPS", border=True)
    else:
        st.info("Метрики ще не пораховані. Запустіть `python evaluate.py`, щоб заповнити дашборд.")

    c1, c2, c3 = st.columns(3)
    with c1, st.container(border=True):
        st.markdown("**:material/neurology: PedNet — власна модель**")
        st.caption("Anchor-free детектор центрів пішоходів. Основа (backbone) — MobileNetV3; власні FPN-шия на stride 4, "
                   "голови heatmap / розміру / зсуву, focal loss і декодування.")
    with c2, st.container(border=True):
        st.markdown("**:material/route: SORT — власний трекер**")
        st.caption("Фільтр Калмана з постійною швидкістю для кожного пішохода та зіставлення детекцій "
                   "з треками за IoU угорським алгоритмом.")
    with c3, st.container(border=True):
        st.markdown("**:material/dataset: Дані**")
        st.caption("Навчання: Caltech Pedestrian (камера на лобовому склі, 640×480). "
                   "Незалежний тест: INRIA Person.")

# ---------------------------------------------------------------- photo
with tabs[1]:
    left, right = st.columns([2, 1], vertical_alignment="bottom")
    with left:
        up = st.file_uploader("Завантажте зображення", type=["jpg", "jpeg", "png", "bmp"], key="photo_up")
    with right:
        b1, b2 = st.columns(2)
        if b1.button("Приклад Caltech", icon=":material/shuffle:", width="stretch"):
            st.session_state.photo_sample = random_sample("Caltech")
        if b2.button("Приклад INRIA", icon=":material/shuffle:", width="stretch"):
            st.session_state.photo_sample = random_sample("INRIA")

    frame, name = None, None
    if up is not None:
        frame, name = decode_upload(up), up.name
    elif st.session_state.get("photo_sample"):
        p = st.session_state.photo_sample
        frame, name = cv2.imread(str(p)), p.name

    if frame is None:
        st.info("Завантажте фото або натисніть «Приклад», щоб узяти випадковий кадр з тестової вибірки.",
                icon=":material/upload:")
    else:
        boxes, ms = detector.timed_predict(frame, conf, iou)
        m1, m2, m3 = st.columns(3)
        m1.metric("Пішоходів знайдено", len(boxes), border=True)
        m2.metric("Макс. впевненість", f"{boxes[:, 4].max():.2f}" if len(boxes) else "—", border=True)
        m3.metric("Час інференсу", f"{ms:.0f} мс", border=True)
        c1, c2 = st.columns(2)
        c1.image(rgb(frame), caption=f"Оригінал · {name}", width="stretch")
        c2.image(rgb(draw_detections(frame, boxes)), caption="Результат детекції", width="stretch")
        with st.expander("Список прямокутників (вихід моделі)", icon=":material/data_array:"):
            tbl = boxes_table(boxes)
            st.dataframe(tbl, hide_index=True, width="stretch")
            st.download_button("Завантажити JSON", json.dumps(tbl.to_dict("records"), indent=1),
                               file_name=f"{Path(name).stem}_boxes.json", mime="application/json",
                               icon=":material/download:")


# ---------------------------------------------------------------- спільний блок живого трекінгу
def live_tracking(frames, total: int | None, fps_out: float, out_name: str, keep_frames: bool = False):
    """Детектор + SORT по всіх кадрах з живим відображенням прогресу та метрик.

    Результат (список прямокутників, відео MP4, статистика) зберігається в st.session_state[out_name],
    щоб він не зникав після натискання кнопок завантаження (Streamlit перезапускає скрипт).
    Відео пишеться кодеком H.264 (libx264), який відтворюється безпосередньо в браузері.
    """
    progress = st.progress(0.0, text="Обробка…")
    k1, k2, k3, k4 = st.columns(4)
    kpi = [k.empty() for k in (k1, k2, k3, k4)]
    view = st.empty()
    tracker = SortTracker(max_age, min_hits)
    tmp = Path(tempfile.mkstemp(suffix=".mp4")[1])
    writer = imageio.get_writer(str(tmp), fps=fps_out, codec="libx264", format="FFMPEG", macro_block_size=1)
    kept, seen_ids, t_start = [], set(), time.time()

    def on_frame(idx, frame, tracks, trk, rec):
        vis = draw_tracks(frame, tracks, trk.trails())
        writer.append_data(rgb(vis))
        seen_ids.update(int(t[5]) for t in tracks)
        if keep_frames and len(kept) < MAX_KEEP_FRAMES:
            kept.append(cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
        # оновлюємо інтерфейс кожен 3-й кадр — частіше лише сповільнює обробку
        if idx % 3 == 0 or (total and idx == total - 1):
            view.image(rgb(vis), width="stretch")
            fps = (idx + 1) / max(time.time() - t_start, 1e-6)
            kpi[0].metric("Кадр", f"{idx + 1}" + (f" / {total}" if total else ""), border=True)
            kpi[1].metric("Пішоходів у кадрі", len(tracks), border=True)
            kpi[2].metric("Унікальних треків", len(seen_ids), border=True)
            kpi[3].metric("Швидкість", f"{fps:.1f} FPS", border=True)
            if total:
                progress.progress(min(1.0, (idx + 1) / total), text=f"Обробка кадру {idx + 1} з {total}")

    results = run_tracking(detector, frames, conf, iou, tracker, on_frame)
    writer.close()
    progress.progress(1.0, text="Готово")
    st.session_state[out_name] = {
        "results": results, "video": tmp.read_bytes(), "frames": kept, "model": info.title,
        "stats": track_stats(results),
    }
    tmp.unlink(missing_ok=True)
    view.empty()


def show_tracking_result(state: dict, prefix: str):
    """Підсумок обробки: картки метрик, графік кількості пішоходів у часі, кнопки завантаження."""
    s = state["stats"]
    st.success(f"Оброблено {s['frames']} кадрів моделлю «{state['model']}».", icon=":material/check_circle:")
    m = st.columns(5)
    m[0].metric("Кадрів", s["frames"], border=True)
    m[1].metric("Треків (пішоходів)", s["tracks"], border=True)
    m[2].metric("Сер. довжина треку", f"{s['avg_track_len']:.1f} кадр.", border=True)
    m[3].metric("Макс. у кадрі", s["max_people"], border=True)
    m[4].metric("Сер. час / кадр", f"{s['avg_ms']:.0f} мс", border=True)

    counts = pd.DataFrame({"кадр": [r["frame"] for r in state["results"]],
                           "пішоходів": [len(r["boxes"]) for r in state["results"]]})
    chart = alt.Chart(counts).mark_area(line={"color": "#2a78d6"}, color="rgba(42,120,214,0.25)").encode(
        x=alt.X("кадр:Q", title="Кадр"), y=alt.Y("пішоходів:Q", title="Пішоходів у кадрі"),
        tooltip=["кадр", "пішоходів"]).properties(height=200)
    st.altair_chart(chart, use_container_width=True)

    df = results_to_frame(state["results"])
    d1, d2, d3 = st.columns(3)
    d1.download_button("Анотоване відео (MP4)", state["video"], file_name=f"{prefix}_tracked.mp4",
                       mime="video/mp4", icon=":material/movie:", width="stretch")
    d2.download_button("Прямокутники (JSON)", json.dumps(state["results"], ensure_ascii=False, indent=1),
                       file_name=f"{prefix}_tracks.json", mime="application/json",
                       icon=":material/data_object:", width="stretch")
    d3.download_button("Прямокутники (CSV)", df.to_csv(index=False), file_name=f"{prefix}_tracks.csv",
                       mime="text/csv", icon=":material/table:", width="stretch")
    return df


# ---------------------------------------------------------------- video
with tabs[2]:
    st.markdown("Завантажте відео з відеореєстратора — кожен пішохід отримає власний ID і траєкторію.")
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    vid = c1.file_uploader("Відео", type=["mp4", "mov", "m4v", "avi"], key="video_up")
    stride = c2.number_input("Обробляти кожен N-й кадр", 1, 10, 1, help="Збільште для швидшої обробки довгих відео.")
    if st.button("Почати обробку", type="primary", icon=":material/play_arrow:", disabled=vid is None):
        in_path = Path(tempfile.mkstemp(suffix=Path(vid.name).suffix)[1])
        in_path.write_bytes(vid.getvalue())
        cap = cv2.VideoCapture(str(in_path))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) // stride or None

        def frames():
            i = 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                if i % stride == 0:
                    yield f"{Path(vid.name).stem}_{i:06d}", f
                i += 1
            cap.release()

        live_tracking(frames(), total, src_fps / stride, "video_state")
        in_path.unlink(missing_ok=True)

    if st.session_state.get("video_state"):
        state = st.session_state.video_state
        st.video(state["video"])
        df = show_tracking_result(state, "video")
        with st.expander("Таблиця прямокутників", icon=":material/table:"):
            st.dataframe(df, hide_index=True, width="stretch", height=300)

# ---------------------------------------------------------------- frame sequence
with tabs[3]:
    st.markdown("Вхід програми за завданням — **послідовність кадрів**, вихід — **список прямокутників** для кожного кадру.")
    src = st.segmented_control("Джерело кадрів", ["Послідовність Caltech (test)", "Мої кадри"],
                               default="Послідовність Caltech (test)", key="seq_src")
    frames_iter, total, prefix = None, None, "frames"
    if src == "Мої кадри":
        files = st.file_uploader("Кадри (кілька зображень) або ZIP-архів", type=["png", "jpg", "jpeg", "zip"],
                                 accept_multiple_files=True, key="seq_up")
        if files:
            items = []
            for f in files:
                if f.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(f.getvalue())) as zf:
                        items += [(Path(n).name, zf.read(n)) for n in zf.namelist()
                                  if Path(n).suffix.lower() in IMG_EXTS and not n.startswith("__MACOSX")]
                else:
                    items.append((f.name, f.getvalue()))
            items.sort(key=lambda x: x[0])
            total = len(items)
            st.caption(f"Кадрів: {total} (впорядковано за назвою файлу)")
            frames_iter = ((n, cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)) for n, b in items)
    else:
        seqs = caltech_sequences()
        if seqs.empty:
            st.warning("Тестова вибірка Caltech не знайдена в `dataset/datasets`.")
        else:
            default = int(seqs.frames.idxmax())
            seq = st.selectbox("Послідовність", seqs.sequence, index=default,
                               format_func=lambda s: f"{s} · {int(seqs.set_index('sequence').frames[s])} кадрів")
            files = sorted((CALTECH_TEST / seq.split("_")[0]).glob(f"{seq}_*.png"))
            total, prefix = len(files), seq
            frames_iter = ((f.name, cv2.imread(str(f))) for f in files)

    if st.button("Обробити послідовність", type="primary", icon=":material/play_arrow:", disabled=frames_iter is None):
        live_tracking(frames_iter, total, 15.0, "seq_state", keep_frames=True)
        st.session_state.seq_prefix = prefix

    if st.session_state.get("seq_state"):
        state = st.session_state.seq_state
        if state["frames"]:
            i = st.slider("Перегляд кадру", 0, len(state["frames"]) - 1, 0, key="seq_view")
            c1, c2 = st.columns([3, 2])
            c1.image(state["frames"][i], width="stretch")
            with c2:
                rec = state["results"][i]
                st.markdown(f"**Кадр {rec['frame']}** · `{rec['file']}`")
                st.json(rec["boxes"], expanded=True)
        show_tracking_result(state, st.session_state.get("seq_prefix", "frames"))

# ---------------------------------------------------------------- comparison
with tabs[4]:
    st.markdown("#### Дві моделі на одному кадрі")
    c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="bottom")
    ma = c1.selectbox("Модель A", models, index=0, format_func=lambda k: MODEL_ZOO[k].title, key="cmp_a")
    mb = c2.selectbox("Модель B", models, index=min(1, len(models) - 1),
                      format_func=lambda k: MODEL_ZOO[k].title, key="cmp_b")
    if c3.button("Новий кадр Caltech", icon=":material/shuffle:", width="stretch") or "cmp_img" not in st.session_state:
        st.session_state.cmp_img = random_sample("Caltech")
    cmp_up = st.file_uploader("…або власне зображення", type=["jpg", "jpeg", "png"], key="cmp_up")
    img = decode_upload(cmp_up) if cmp_up else (cv2.imread(str(st.session_state.cmp_img))
                                                if st.session_state.cmp_img else None)
    if img is not None:
        cols = st.columns(2)
        for col, key in zip(cols, (ma, mb)):
            det = get_detector(key)
            bx, ms = det.timed_predict(img, conf, iou)
            with col:
                st.markdown(f"**{MODEL_ZOO[key].title}** {badge(MODEL_ZOO[key].origin)}", unsafe_allow_html=True)
                st.image(rgb(draw_detections(img, bx)), width="stretch")
                st.caption(f"Знайдено: **{len(bx)}** · час: **{ms:.0f} мс**")

    st.divider()
    st.markdown("#### Метрики на тестових вибірках")
    df_m = load_metrics()
    if df_m is None:
        st.info("Запустіть `python evaluate.py`, щоб побачити таблиці та криві.")
    else:
        ds = st.segmented_control("Датасет", ["Caltech", "INRIA"], default="Caltech", key="cmp_ds") or "Caltech"
        view = df_m[(df_m.dataset == ds)].copy()
        view["Модель"] = view.model.map(MODEL_SHORT)
        view["Тип"] = view.origin.map(ORIGIN_LABELS)
        view["MR⁻² (%)"] = view.mr2 * 100
        st.dataframe(
            view[["Модель", "Тип", "subset", "precision", "recall", "f1", "ap50", "ap50_95", "MR⁻² (%)",
                  "fps", "params_m", "size_mb"]],
            hide_index=True, width="stretch",
            column_config={
                "subset": st.column_config.TextColumn("Протокол", help="all — усі детекції; h60 — детекції нижчі за 60 px ігноруються (розмічено лише пішоходів ≥ 75 px)"),
                "precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                "recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                "f1": st.column_config.NumberColumn("F1", format="%.3f"),
                "ap50": st.column_config.ProgressColumn("AP50", format="%.3f", min_value=0, max_value=1),
                "ap50_95": st.column_config.NumberColumn("AP50-95", format="%.3f"),
                "MR⁻² (%)": st.column_config.NumberColumn("MR⁻² (%)", format="%.1f",
                                                          help="Log-average miss rate (менше — краще)"),
                "fps": st.column_config.NumberColumn("FPS", format="%.0f"),
                "params_m": st.column_config.NumberColumn("Параметрів, M", format="%.2f"),
                "size_mb": st.column_config.NumberColumn("Розмір, МБ", format="%.1f"),
            },
        )
        curves = load_curves()
        if curves:
            pr, mr = [], []
            for m in MODEL_ZOO:
                k = f"{m}|{ds}"
                if f"{k}|rec" not in curves:
                    continue
                pr.append(pd.DataFrame({"Recall": curves[f"{k}|rec"], "Precision": curves[f"{k}|prec"],
                                        "Модель": MODEL_SHORT[m]}))
                mr.append(pd.DataFrame({"FPPI": curves[f"{k}|fppi"], "Miss rate": curves[f"{k}|miss"],
                                        "Модель": MODEL_SHORT[m]}))
            dom = [MODEL_SHORT[m] for m in MODEL_ZOO]
            rng = [MODEL_COLORS[m] for m in MODEL_ZOO]
            color = alt.Color("Модель:N", scale=alt.Scale(domain=dom, range=rng), legend=alt.Legend(orient="bottom"))
            g1, g2 = st.columns(2)
            with g1:
                st.markdown("**Precision–Recall** (IoU ≥ 0.5)")
                st.altair_chart(alt.Chart(pd.concat(pr)).mark_line(strokeWidth=2.5).encode(
                    x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1])), color=color,
                    tooltip=["Модель", alt.Tooltip("Recall:Q", format=".3f"), alt.Tooltip("Precision:Q", format=".3f")]
                ).properties(height=330), use_container_width=True)
            with g2:
                st.markdown("**Miss rate – FPPI** (протокол Caltech, менше — краще)")
                d = pd.concat(mr)
                d = d[(d.FPPI >= 1e-2) & (d.FPPI <= 10)]
                st.altair_chart(alt.Chart(d).mark_line(strokeWidth=2.5).encode(
                    x=alt.X("FPPI:Q", scale=alt.Scale(type="log", domain=[0.01, 10]), title="Хибних спрацювань на кадр"),
                    y=alt.Y("Miss rate:Q", scale=alt.Scale(type="log", domain=[0.05, 1])), color=color,
                    tooltip=["Модель", alt.Tooltip("FPPI:Q", format=".3f"), alt.Tooltip("Miss rate:Q", format=".3f")]
                ).properties(height=330), use_container_width=True)

# ---------------------------------------------------------------- training
with tabs[5]:
    hist = Path("runs/pednet/history.csv")
    yolo_res = sorted(Path("runs").rglob("caltech_v8n*/results.csv"), key=lambda f: f.stat().st_mtime)[-1:]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### PedNet — власний цикл навчання")
        if hist.exists():
            h = pd.read_csv(hist)
            last = h.iloc[-1]
            m = st.columns(3)
            m[0].metric("Епох", int(last.epoch), border=True)
            m[1].metric("Найкращий val AP50", f"{h.val_ap50.max():.3f}", border=True)
            m[2].metric("Час епохи", f"{h.time_s.mean() / 60:.1f} хв", border=True)
            long = h.melt("epoch", ["hm", "size", "off"], var_name="компонента", value_name="loss")
            st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
                x="epoch:Q", y=alt.Y("loss:Q", title="Loss"), color=alt.Color("компонента:N", legend=alt.Legend(orient="bottom")),
                tooltip=["epoch", "компонента", alt.Tooltip("loss:Q", format=".3f")]).properties(height=240),
                use_container_width=True)
            st.altair_chart(alt.Chart(h).mark_line(point=True, color="#2a78d6").encode(
                x="epoch:Q", y=alt.Y("val_ap50:Q", title="val AP50"),
                tooltip=["epoch", alt.Tooltip("val_ap50:Q", format=".3f")]).properties(height=200),
                use_container_width=True)
        else:
            st.info("Історія навчання PedNet з'явиться після запуску `python train_pednet.py`.")
    with c2:
        st.markdown("#### YOLOv8n — fine-tuning (Ultralytics)")
        if yolo_res:
            y = pd.read_csv(yolo_res[0])
            y.columns = [c.strip() for c in y.columns]
            m = st.columns(3)
            m[0].metric("Епох", int(y.epoch.max()), border=True)
            m[1].metric("Найкращий val mAP50", f"{y['metrics/mAP50(B)'].max():.3f}", border=True)
            m[2].metric("mAP50-95", f"{y['metrics/mAP50-95(B)'].max():.3f}", border=True)
            long = y.melt("epoch", ["train/box_loss", "train/cls_loss", "train/dfl_loss"],
                          var_name="компонента", value_name="loss")
            st.altair_chart(alt.Chart(long).mark_line(point=True).encode(
                x="epoch:Q", y="loss:Q", color=alt.Color("компонента:N", legend=alt.Legend(orient="bottom")),
                tooltip=["epoch", "компонента", alt.Tooltip("loss:Q", format=".3f")]).properties(height=240),
                use_container_width=True)
            st.altair_chart(alt.Chart(y).mark_line(point=True, color="#e0702b").encode(
                x="epoch:Q", y=alt.Y("metrics/mAP50(B):Q", title="val mAP50"),
                tooltip=["epoch", alt.Tooltip("metrics/mAP50(B):Q", format=".3f")]).properties(height=200),
                use_container_width=True)
        else:
            st.info("Результати навчання YOLO з'являться після `python train_yolo.py`.")

    tr = REPORTS / "tracking.csv"
    if tr.exists():
        st.markdown("#### Трекінг на послідовностях Caltech test")
        t = pd.read_csv(tr)
        t["model"] = t.model.map(MODEL_SHORT)
        st.dataframe(t, hide_index=True, width="stretch",
                     column_config={"avg_track_len": st.column_config.NumberColumn("Сер. довжина треку", format="%.1f"),
                                    "avg_people": st.column_config.NumberColumn("Сер. пішоходів/кадр", format="%.2f"),
                                    "avg_ms": st.column_config.NumberColumn("мс/кадр", format="%.1f"),
                                    "fps": st.column_config.NumberColumn("FPS", format="%.1f")})
