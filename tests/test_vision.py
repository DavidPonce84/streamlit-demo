from io import BytesIO

from PIL import Image

from src.vision import classify_image, load_image, torchvision_available


def image_bytes(color=(120, 120, 120)):
    image = Image.new("RGB", (32, 32), color=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_load_image_converts_to_rgb():
    image = load_image(image_bytes())
    assert image.mode == "RGB"
    assert image.size == (32, 32)


def test_classifier_returns_bounded_probabilities_baseline():
    result = classify_image(load_image(image_bytes()), force_baseline=True)
    assert result["label"]
    assert 0 <= result["confidence"] <= 1
    assert abs(sum(result["probabilities"].values()) - 1) < 1e-4
    assert result["baseline"] is True


def test_classifier_returns_bounded_probabilities_mobilenet():
    if torchvision_available():
        result = classify_image(load_image(image_bytes()), force_baseline=False)
        assert result["label"]
        assert 0 <= result["confidence"] <= 1
        assert abs(sum(result["probabilities"].values()) - 1) < 1e-4
        assert result["baseline"] is False
        assert "MobileNetV2" in result["model_name"]


def test_classifier_is_deterministic():
    image = load_image(image_bytes((250, 250, 250)))
    assert classify_image(image, force_baseline=True) == classify_image(image, force_baseline=True)
