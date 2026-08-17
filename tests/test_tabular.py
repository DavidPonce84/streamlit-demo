import pandas as pd
import pytest

from src.tabular import FEATURES, build_demo_dataset, dataframe_from_csv, predict_tabular, train_tabular_model


def test_demo_dataset_is_deterministic_and_valid():
    a = build_demo_dataset(100)
    b = build_demo_dataset(100)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns) == FEATURES + ["renovara"]
    assert len(a) == 100


def test_training_and_prediction_shape():
    model, X_train, X_test, y_train, y_test = train_tabular_model(build_demo_dataset())
    result = predict_tabular(model, X_test.head(7))
    assert len(result) == 7
    assert "prediccion" in result
    assert result["probabilidad_renovacion"].between(0, 1).all()


def test_missing_columns_are_rejected():
    model, *_ = train_tabular_model(build_demo_dataset())
    with pytest.raises(ValueError, match="Faltan columnas"):
        predict_tabular(model, pd.DataFrame({"ingreso_mensual": [1]}))


def test_csv_parser_validates_schema():
    frame = build_demo_dataset(20).head(3).drop(columns=["renovara"])
    parsed = dataframe_from_csv(frame.to_csv(index=False).encode())
    assert list(parsed.columns) == FEATURES
    with pytest.raises(ValueError, match="Faltan columnas"):
        dataframe_from_csv(b"ingreso_mensual\n10\n")
