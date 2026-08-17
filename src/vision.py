"""Image classifier supporting pretrained MobileNetV2 and offline heuristic baseline."""
from __future__ import annotations

from io import BytesIO
from typing import BinaryIO

import numpy as np
from PIL import Image

LABELS = ["avion", "automóvil", "pájaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión"]

_MODEL_CACHE = None


def torchvision_available() -> bool:
    """Check if PyTorch and Torchvision are installed and importable."""
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
        return True
    except Exception:
        return False


def load_image(source: bytes | BinaryIO | Image.Image) -> Image.Image:
    """Load and normalize an image source."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, bytes):
        return Image.open(BytesIO(source)).convert("RGB")
    return Image.open(source).convert("RGB")


def _get_mobilenet_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        import torchvision.models as models
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
        model.eval()
        _MODEL_CACHE = (model, weights)
    return _MODEL_CACHE


def classify_image_mobilenet(image: Image.Image) -> dict:
    """Classify an image using PyTorch pretrained MobileNetV2."""
    import torch

    model, weights = _get_mobilenet_model()
    preprocess = weights.transforms()
    img_tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor).squeeze(0)
        probabilities_tensor = torch.nn.functional.softmax(output, dim=0)

    top5_prob, top5_catid = torch.topk(probabilities_tensor, 5)
    top5_norm = top5_prob / top5_prob.sum()
    categories = weights.meta["categories"]

    label = categories[top5_catid[0].item()]
    confidence = float(top5_prob[0].item())

    probabilities = {
        categories[top5_catid[i].item()]: float(top5_norm[i].item())
        for i in range(5)
    }

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "baseline": False,
        "model_name": "MobileNetV2 (PyTorch ImageNet)"
    }


def classify_image(image: Image.Image, force_baseline: bool = False) -> dict:
    """Return an image classification result using MobileNetV2 if available, or offline fallback baseline."""
    if not force_baseline and torchvision_available():
        try:
            return classify_image_mobilenet(image)
        except Exception:
            pass

    # Heuristic baseline fallback
    image_resized = image.convert("RGB").resize((32, 32))
    arr = np.asarray(image_resized, dtype=np.float32) / 255.0
    brightness = float(arr.mean())
    saturation = float(arr.max(axis=2).mean() - arr.min(axis=2).mean())

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

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "baseline": True,
        "model_name": "Baseline didáctico offline"
    }
