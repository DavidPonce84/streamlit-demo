"""Train and serialize the deterministic tabular demo model."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from src.tabular import build_demo_dataset, evaluate_tabular, train_tabular_model

import joblib

out = ROOT / "models"
out.mkdir(exist_ok=True)
frame = build_demo_dataset()
model, _, X_test, _, y_test = train_tabular_model(frame)
artifact = out / "random_forest_renewal.joblib"
joblib.dump(model, artifact)
metadata = {
    "model_type": "RandomForestClassifier",
    "target": "renovara",
    "features": ["ingreso_mensual", "antiguedad_meses", "tickets_previos", "satisfaccion", "tiempo_respuesta"],
    "random_state": 42,
    "training_rows": len(frame),
    "metrics_holdout": evaluate_tabular(model, X_test, y_test),
}
(out / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Artefacto creado: {artifact}")
print(json.dumps(metadata, indent=2, ensure_ascii=False))
