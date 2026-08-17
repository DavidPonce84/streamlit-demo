"""Lightweight image classifier for the live educational demo."""
from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image

LABELS = ["avion", "automóvil", "pájaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión"]


def load_image(source: bytes | BinaryIO | Image.Image) -> Image.Image:
    """Load and normalize an image source."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, bytes):
        return Image.open(BytesIO(source)).convert("RGB")
    return Image.open(source).convert("RGB")


def classify_image(image: Image.Image) -> dict:
    """Return a deterministic, offline-safe image classification result.

    The showcase intentionally exposes a lightweight baseline rather than
    downloading weights during class. A real pretrained model can replace
    this function without changing the Streamlit interface.
    """
    image = image.convert("RGB").resize((32, 32))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    brightness = float(arr.mean())
    saturation = float(arr.max(axis=2).mean() - arr.min(axis=2).mean())
    # Educational heuristic baseline: transparent about its limitations.
    if brightness > 0.72 and saturation < 0.18:
        label = "avión"
    elif saturation > 0.38:
        label = "pájaro"
    else:
        label = "gato"
    confidence = float(np.clip(0.45 + abs(brightness - 0.5) * 0.25 + saturation * 0.2, 0.45, 0.78))
    top = {name: max(0.01, confidence - i * 0.1) for i, name in enumerate([label, "perro", "caballo"])}
    total = sum(top.values())
    probabilities = {name: value / total for name, value in top.items()}
    return {"label": label, "confidence": confidence, "probabilities": probabilities, "baseline": True}
