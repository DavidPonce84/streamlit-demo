import pandas as pd
from src.ui_components import (
    create_donut_gauge,
    create_global_importance_chart,
    create_local_explanation_chart,
    create_vision_probabilities_chart,
)


def test_global_importance_chart_creation():
    df = pd.DataFrame({"feature": ["f1", "f2"], "importance": [0.6, 0.4]})
    chart = create_global_importance_chart(df, "SHAP")
    assert chart is not None
    assert chart.to_dict() is not None


def test_local_explanation_chart_creation():
    df = pd.DataFrame({"feature": ["f1", "f2"], "impact": [0.15, -0.05]})
    chart = create_local_explanation_chart(df, "LIME")
    assert chart is not None
    assert chart.to_dict() is not None


def test_vision_probabilities_chart_creation():
    probs = {"cat": 0.8, "dog": 0.2}
    chart = create_vision_probabilities_chart(probs, "cat")
    assert chart is not None
    assert chart.to_dict() is not None


def test_donut_gauge_creation():
    chart = create_donut_gauge(0.75)
    assert chart is not None
    assert chart.to_dict() is not None
