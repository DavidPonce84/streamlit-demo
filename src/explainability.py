"""Optional SHAP/LIME explanations with deterministic fallbacks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tabular import FEATURES, permutation_importances


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
        return True
    except Exception:
        return False


def lime_available() -> bool:
    try:
        import lime  # noqa: F401
        return True
    except Exception:
        return False


def global_importance(model, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, str]:
    """Return global importance and method used; SHAP when installed, fallback otherwise."""
    if shap_available():
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X_test)
            if isinstance(values, list):
                values = values[1]
            importance = np.abs(np.asarray(values)).mean(axis=0)
            result = pd.DataFrame({"feature": FEATURES, "importance": importance})
            return result.sort_values("importance", ascending=False), "SHAP"
        except Exception:
            pass
    return permutation_importances(model, X_test, y_test), "Permutation importance (fallback)"


def local_explanation(model, row: pd.DataFrame, X_background: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Explain one row. Uses SHAP, LIME, or a model-native fallback."""
    if shap_available():
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(row)
            if isinstance(values, list):
                values = values[1]
            result = pd.DataFrame({"feature": FEATURES, "impact": np.asarray(values)[0]})
            return result.sort_values("impact", key=np.abs, ascending=False), "SHAP local"
        except Exception:
            pass
    if lime_available():
        try:
            from lime.lime_tabular import LimeTabularExplainer
            explainer = LimeTabularExplainer(
                X_background.to_numpy(), feature_names=FEATURES, class_names=["No", "Si"], mode="classification", random_state=42
            )
            explanation = explainer.explain_instance(row.iloc[0].to_numpy(), model.predict_proba, num_features=len(FEATURES))
            result = pd.DataFrame(explanation.as_list(), columns=["feature", "impact"])
            return result, "LIME local"
        except Exception:
            pass
    # Transparent fallback: signed perturbation estimate.
    base = float(model.predict_proba(row)[0, 1])
    impacts = []
    for feature in FEATURES:
        changed = row.copy()
        changed[feature] = X_background[feature].median()
        impacts.append(base - float(model.predict_proba(changed)[0, 1]))
    return pd.DataFrame({"feature": FEATURES, "impact": impacts}).sort_values("impact", key=np.abs, ascending=False), "Perturbation fallback"
