from io import BytesIO

from PIL import Image

from src.vision import classify_image, load_image


def image_bytes(color=(120, 120, 120)):
    image = Image.new("RGB", (32, 32), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_converts_to_rgb():
    image = load_image(image_bytes())
    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_classifier_returns_bounded_probabilities():
    result = classify_image(load_image(image_bytes()))
    assert result["label"]
    assert 0 <= result["confidence"] <= 1
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-9
    assert result["baseline"] is True


def test_classifier_is_deterministic():
    image = load_image(image_bytes((250, 250, 250)))
    assert classify_image(image) == classify_image(image)
