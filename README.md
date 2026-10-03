# PedVision — виявлення та відстеження пішоходів
### Ходаков Максим Олегович, ШІ-2

Система локалізує та відстежує пішоходів на відео з камери на лобовому склі автомобіля.
**Вхід:** послідовність кадрів (папка зображень / відео). **Вихід:** список прямокутників `track_id, x1, y1, x2, y2, conf` для кожного кадру.

Моделі:
* **PedNet** (`pedestrian/pednet.py`) — власний anchor-free детектор (MobileNetV3 + власні FPN, голови heatmap/size/offset, focal loss), навчений у проєкті;
* **YOLOv8n**, донавчена на Caltech (`train_yolo.py`), а також базова YOLOv8n COCO для порівняння;
* **SORT** (`pedestrian/sort_tracker.py`) — власна реалізація трекера (Kalman + Hungarian).

Дані: Caltech Pedestrian (навчання/валідація/тест), INRIA Person (незалежний тест).

## Запуск
```bash
pip install -r requirements.txt

python prepare_inria.py                  # INRIA -> YOLO-формат (dataset/inria)
python train_yolo.py --epochs 20         # fine-tune YOLOv8n  -> models/yolo_caltech_v8n.pt
python train_pednet.py --epochs 25       # навчання PedNet     -> models/pednet_best.pt
python evaluate.py                       # метрики та графіки  -> reports/

# трекінг послідовності кадрів -> JSON/CSV (+ відео)
python track.py --source dataset/datasets/images/test/caltechpedestriandataset/set06 \
                --pattern "set06_V016_*" --model pednet --out outputs/set06_V016.json --save-video

python -m streamlit run app.py           # веб-платформа
```

## Структура
```
pedestrian/   data.py · pednet.py · sort_tracker.py · detectors.py · metrics.py · viz.py
train_pednet.py · train_yolo.py · evaluate.py · track.py · prepare_inria.py · app.py
models/       ваги моделей          reports/   метрики, криві, приклади
```
