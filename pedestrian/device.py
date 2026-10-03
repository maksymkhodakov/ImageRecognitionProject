"""Вибір обчислювального пристрою для PyTorch."""
from __future__ import annotations

import os

import torch


def pick_device(prefer: str | None = None) -> str:
    """Повертає найкращий доступний пристрій: cuda (NVIDIA) -> mps (Apple Silicon) -> cpu.

    Пріоритет: явно переданий prefer > змінна середовища PED_DEVICE > автовизначення.
    PED_DEVICE=cpu корисна, коли GPU зайнятий навчанням, а потрібно щось швидко перевірити.
    """
    if prefer:
        return prefer
    if os.environ.get("PED_DEVICE"):
        return os.environ["PED_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
