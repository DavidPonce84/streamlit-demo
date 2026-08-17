"""Safe model artifact loading and metadata for the live showcase."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .tabular import build_demo_dataset, train_tabular_model


ARTIFACT = Path(__file__).parents[1] / "models" / "random_forest_renewal.joblib"
METADATA = Path(__file__).parents[1] / "models" / "model_metadata.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_artifact():
    """Load a local artifact if valid; otherwise deterministic training fallback."""
    if ARTIFACT.exists():
        try:
            import joblib
            model = joblib.load(ARTIFACT)
            if METADATA.exists():
                metadata = json.loads(METADATA.read_text(encoding="utf-8"))
            else:
                metadata = {}
            return model, {"source": str(ARTIFACT.relative_to(ARTIFACT.parents[1])), "artifact": True, **metadata}
        except Exception as exc:
            reason = f"artefacto no cargable: {type(exc).__name__}"
    else:
        reason = "artefacto ausente; se entrenó fallback determinístico"
    model, *_ = train_tabular_model(build_demo_dataset())
    return model, {"source": "entrenamiento local reproducible", "artifact": False, "reason": reason, "random_state": 42}
