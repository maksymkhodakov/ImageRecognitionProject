from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> str:
    """Return the best available torch device: cuda -> mps -> cpu."""
    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
