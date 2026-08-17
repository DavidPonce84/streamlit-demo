"""Verify model artifact integrity and metadata."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.model_registry import ARTIFACT, METADATA, sha256_file

if not ARTIFACT.exists():
    raise SystemExit("Falta models/random_forest_renewal.joblib; ejecute scripts/train_model.py")
if not METADATA.exists():
    raise SystemExit("Falta models/model_metadata.json")
metadata = json.loads(METADATA.read_text(encoding="utf-8"))
required = {"model_type", "features", "random_state", "metrics_holdout"}
missing = required - set(metadata)
if missing:
    raise SystemExit(f"Metadatos incompletos: {sorted(missing)}")
print(f"OK artefacto: {ARTIFACT}")
print(f"SHA256: {sha256_file(ARTIFACT)}")
print(json.dumps(metadata, indent=2, ensure_ascii=False))
