"""Пакет PedVision: виявлення та відстеження пішоходів.

Модулі (у порядку проходження даних):
    data.py          читання датасетів у форматі YOLO, letterbox, нормалізація, аугментації, цілі PedNet
    pednet.py        власна модель PedNet: архітектура, функція втрат, декодування виходів
    detectors.py     єдиний інтерфейс Detector для PedNet і YOLO + реєстр моделей MODEL_ZOO
    sort_tracker.py  трекер SORT (фільтр Калмана + угорський алгоритм)
    metrics.py       IoU, Precision/Recall, AP, log-average miss rate
    viz.py           малювання рамок, ID і траєкторій
    device.py        вибір пристрою cuda / mps / cpu
"""
