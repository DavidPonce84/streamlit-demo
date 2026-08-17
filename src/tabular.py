"""Tabular classification demo with deterministic data and model."""
from __future__ import annotations

from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

FEATURES = ["ingreso_mensual", "antiguedad_meses", "tickets_previos", "satisfaccion", "tiempo_respuesta"]
TARGET = "renovara"


def build_demo_dataset(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Build a synthetic customer-renewal dataset for live demonstrations."""
    if n_samples < 20:
        raise ValueError("n_samples debe ser al menos 20")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(FEATURES),
        n_informative=4,
        n_redundant=1,
        weights=[0.42, 0.58],
        class_sep=1.05,
        random_state=random_state,
    )
    # Scale to values that make sense in the dashboard.
    frame = pd.DataFrame(X, columns=FEATURES)
    frame["ingreso_mensual"] = np.round(np.exp(frame["ingreso_mensual"] / 2 + 7), 2)
    frame["antiguedad_meses"] = np.clip(np.round((frame["antiguedad_meses"] + 2.5) * 14), 1, 120).astype(int)
    frame["tickets_previos"] = np.clip(np.round((frame["tickets_previos"] + 2) * 2), 0, 20).astype(int)
    frame["satisfaccion"] = np.clip(np.round((frame["satisfaccion"] + 2.5) * 2 + 5), 1, 10).astype(int)
    frame["tiempo_respuesta"] = np.clip(np.round((frame["tiempo_respuesta"] + 2.5) * 3 + 8), 1, 30).astype(int)
    frame[TARGET] = y
    return frame


def train_tabular_model(frame: pd.DataFrame, random_state: int = 42):
    """Train a reproducible Random Forest and return model plus holdout data."""
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    X, y = frame[FEATURES], frame[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )
    model = RandomForestClassifier(
        n_estimators=120, max_depth=6, min_samples_leaf=3, random_state=random_state, n_jobs=1
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def predict_tabular(model, frame: pd.DataFrame) -> pd.DataFrame:
    """Return class predictions and probabilities for one or many rows."""
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    X = frame[FEATURES].copy()
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    result = frame.copy()
    result["prediccion"] = pred.astype(int)
    result["probabilidad_renovacion"] = np.round(prob, 4)
    return result


def csv_template() -> bytes:
    """Return a CSV template suitable for the upload widget."""
    return build_demo_dataset(5).drop(columns=[TARGET]).to_csv(index=False).encode("utf-8")


def evaluate_tabular(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute stable metrics for the dashboard."""
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
        "auc": float(roc_auc_score(y_test, proba)),
    }


def permutation_importances(model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Model-agnostic global importance fallback that needs no SHAP dependency."""
    result = permutation_importance(model, X_test, y_test, n_repeats=8, random_state=42, n_jobs=1)
    return pd.DataFrame({"feature": FEATURES, "importance": result.importances_mean}).sort_values("importance", ascending=False)


def dataframe_from_csv(raw: bytes) -> pd.DataFrame:
    """Parse an uploaded CSV and validate its feature schema."""
    frame = pd.read_csv(BytesIO(raw))
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"CSV inválido. Faltan columnas: {sorted(missing)}")
    return frame
