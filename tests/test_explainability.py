import pandas as pd

from src.explainability import global_importance, local_explanation, lime_available, shap_available
from src.tabular import FEATURES, build_demo_dataset, train_tabular_model


def test_global_importance_has_all_features():
    model, _, X_test, _, y_test = train_tabular_model(build_demo_dataset())
    result, method = global_importance(model, X_test, y_test)
    assert set(result["feature"]) == set(FEATURES)
    assert method
    assert len(result) == len(FEATURES)


def test_local_explanation_returns_one_row_per_feature_or_lime_items():
    model, X_train, _, _, _ = train_tabular_model(build_demo_dataset())
    result, method = local_explanation(model, X_train.iloc[[0]], X_train)
    assert len(result) >= len(FEATURES) if "LIME" not in method else len(result) > 0
    assert "impact" in result.columns
    assert method


def test_optional_flags_are_boolean():
    assert isinstance(shap_available(), bool)
    assert isinstance(lime_available(), bool)
