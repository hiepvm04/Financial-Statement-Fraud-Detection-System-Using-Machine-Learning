from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.train import (
    split_by_year,
    prepare_xy,
    evaluate_binary_classification,
    get_model_predictions,
    evaluate_on_split,
    tune_threshold,
)
from src.evaluate import (
    get_confusion_matrix_df,
    get_classification_report_df,
    summarize_predictions,
)


def make_model_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)

    rows = []
    years = [2019, 2020, 2021, 2022, 2023]

    for cp in ["AAA", "BBB", "CCC", "DDD", "EEE"]:
        for year in years:
            rows.append(
                {
                    "CP": cp,
                    "Năm": year,
                    "DSRI": float(rng.normal(1.0, 0.1)),
                    "GMI": float(rng.normal(1.0, 0.1)),
                    "AQI": float(rng.normal(1.0, 0.1)),
                    "SGI": float(rng.normal(1.1, 0.15)),
                    "DEPI": float(rng.normal(1.0, 0.1)),
                    "SGAI": float(rng.normal(1.0, 0.1)),
                    "LVGI": float(rng.normal(1.0, 0.1)),
                    "TATA": float(rng.normal(0.0, 0.05)),
                    "RSST_Accruals": float(rng.normal(0.0, 0.05)),
                    "Delta_Receivables": float(rng.normal(0.0, 0.03)),
                    "Delta_Inventory": float(rng.normal(0.0, 0.03)),
                    "Delta_Cash_Sales": float(rng.normal(0.0, 0.10)),
                    "Fraud": int(rng.uniform() > 0.7),
                }
            )

    return pd.DataFrame(rows)


def train_simple_model(df: pd.DataFrame):
    train_df = df[df["Năm"] <= 2021].copy()
    test_df = df[df["Năm"] >= 2022].copy()

    X_train, y_train, feature_cols = prepare_xy(train_df)
    X_test, y_test, _ = prepare_xy(test_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_test, X_test_scaled, y_test, feature_cols, test_df


def test_split_by_year_returns_expected_splits():
    df = make_model_dataset()

    train_df, valid_df, test_df = split_by_year(
        df, train_end_year=2021, valid_year=2022, test_year=2023
    )

    assert (train_df["Năm"] <= 2021).all()
    assert (valid_df["Năm"] == 2022).all()
    assert (test_df["Năm"] == 2023).all()


def test_prepare_xy_returns_correct_feature_set():
    df = make_model_dataset()

    X, y, feature_cols = prepare_xy(df)

    assert "CP" not in feature_cols
    assert "Năm" not in feature_cols
    assert "Fraud" not in feature_cols
    assert X.shape[1] == len(feature_cols)
    assert y.name == "Fraud"


def test_get_model_predictions_returns_valid_shapes_and_probabilities():
    df = make_model_dataset()
    model, scaler, X_test, X_test_scaled, y_test, _, _ = train_simple_model(df)

    y_pred, y_prob = get_model_predictions(model, X_test_scaled)

    assert len(y_pred) == len(y_test)
    assert len(y_prob) == len(y_test)
    assert np.all((y_prob >= 0) & (y_prob <= 1))


def test_evaluate_binary_classification_returns_expected_metrics():
    y_true = pd.Series([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.4, 0.2, 0.8])

    metrics = evaluate_binary_classification(y_true, y_pred, y_prob)

    expected_keys = {"Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"}
    assert expected_keys.issubset(metrics.keys())


def test_evaluate_on_split_returns_split_name():
    df = make_model_dataset()
    model, scaler, X_test, X_test_scaled, y_test, _, _ = train_simple_model(df)

    result = evaluate_on_split(model, X_test_scaled, y_test, split_name="Test")

    assert result["Split"] == "Test"
    assert "Accuracy" in result
    assert "F1" in result


def test_tune_threshold_returns_best_row_and_dataframe():
    y_true = pd.Series([0, 1, 1, 0, 1, 0, 1, 0])
    y_prob = np.array([0.1, 0.8, 0.6, 0.3, 0.7, 0.2, 0.9, 0.4])

    threshold_df, best_row = tune_threshold(y_true, y_prob, metric="f1")

    assert not threshold_df.empty
    assert "threshold" in threshold_df.columns
    assert "f1" in threshold_df.columns
    assert 0.1 <= best_row["threshold"] <= 0.9


def test_evaluate_helpers_return_valid_dataframes():
    df = make_model_dataset()
    model, scaler, X_test, X_test_scaled, y_test, feature_cols, test_df = train_simple_model(df)

    y_pred, y_prob = get_model_predictions(model, X_test_scaled)

    cm_df = get_confusion_matrix_df(y_test, y_pred)
    report_df = get_classification_report_df(y_test, y_pred)
    pred_df = summarize_predictions(test_df, y_test, y_pred, y_prob)

    assert cm_df.shape == (2, 2)
    assert not report_df.empty
    assert {"CP", "Năm", "Actual", "Predicted", "Predicted_Prob"}.issubset(pred_df.columns)