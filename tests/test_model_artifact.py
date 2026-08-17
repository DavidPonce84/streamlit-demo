import json

from src.model_registry import ARTIFACT, METADATA, load_model_artifact, sha256_file


def test_model_artifact_is_loadable_and_metadata_is_present():
    model, info = load_model_artifact()
    assert model is not None
    assert "source" in info
    if ARTIFACT.exists():
        assert info["artifact"] is True
        metadata = json.loads(METADATA.read_text(encoding="utf-8"))
        assert metadata["model_type"] == "RandomForestClassifier"
        assert len(sha256_file(ARTIFACT)) == 64


def test_artifact_has_no_secret_like_content():
    if ARTIFACT.exists():
        raw = ARTIFACT.read_bytes()
        assert b"sk-" not in raw
        assert b"OPENCODE_API_KEY" not in raw
