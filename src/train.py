from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE,
    TARGET_COL,
    TRAIN_END_YEAR,
    VALID_YEAR,
    TEST_YEAR,
)


def load_model_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def split_by_year(
    df: pd.DataFrame,
    train_end_year: int = TRAIN_END_YEAR,
    valid_year: int = VALID_YEAR,
    test_year: int = TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["Năm"] <= train_end_year].copy()
    valid_df = df[df["Năm"] == valid_year].copy()
    test_df = df[df["Năm"] == test_year].copy()
    return train_df, valid_df, test_df


def prepare_xy(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    exclude_cols = exclude_cols or ["CP", "Năm", target_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, feature_cols


def evaluate_binary_classification(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
    }


def get_model_predictions(model: Any, X: Any) -> tuple[np.ndarray, np.ndarray]:
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)
        y_prob = 1 / (1 + np.exp(-scores))

    return y_pred, y_prob


def evaluate_on_split(model: Any, X: Any, y: pd.Series, split_name: str = "Validation") -> dict:
    y_pred, y_prob = get_model_predictions(model, X)
    result = evaluate_binary_classification(y, y_pred, y_prob)
    result["Split"] = split_name
    return result


def manual_tune_model(
    model_class: Any,
    param_grid: dict,
    X_train: Any,
    y_train: pd.Series,
    X_valid: Any,
    y_valid: pd.Series,
    fixed_params: dict | None = None,
) -> tuple[Any, dict, pd.DataFrame]:
    fixed_params = fixed_params or {}
    results = []
    best_model = None
    best_params = None
    best_score = -np.inf

    for params in ParameterGrid(param_grid):
        full_params = {**fixed_params, **params}
        model = model_class(**full_params)
        model.fit(X_train, y_train)

        y_pred, y_prob = get_model_predictions(model, X_valid)
        metrics = evaluate_binary_classification(y_valid, y_pred, y_prob)

        row = {**full_params, **metrics}
        results.append(row)

        if metrics["ROC_AUC"] > best_score:
            best_score = metrics["ROC_AUC"]
            best_model = deepcopy(model)
            best_params = full_params

    results_df = pd.DataFrame(results).sort_values(
        by=["ROC_AUC", "PR_AUC", "F1"], ascending=False
    ).reset_index(drop=True)

    return best_model, best_params, results_df


def get_default_param_grids(y_train: pd.Series) -> dict[str, dict]:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    return {
        "logistic_regression": {
            "model_class": LogisticRegression,
            "param_grid": {
                "C": [0.01, 0.1, 1, 5, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs"],
                "class_weight": [None, "balanced"],
                "max_iter": [1000],
            },
            "fixed_params": {},
            "use_scaled": True,
        },
        "xgboost": {
            "model_class": XGBClassifier,
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5],
                "learning_rate": [0.03, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            },
            "fixed_params": {
                "random_state": RANDOM_STATE,
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "scale_pos_weight": scale_pos_weight,
            },
            "use_scaled": False,
        },
        "lightgbm": {
            "model_class": LGBMClassifier,
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.03, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 3, 5],
                "subsample": [0.8, 1.0],
            },
            "fixed_params": {
                "random_state": RANDOM_STATE,
                "class_weight": "balanced",
                "verbosity": -1,
            },
            "use_scaled": False,
        },
        "ann": {
            "model_class": MLPClassifier,
            "param_grid": {
                "hidden_layer_sizes": [(32,), (64,), (64, 32)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01],
                "max_iter": [500],
            },
            "fixed_params": {
                "random_state": RANDOM_STATE,
                "early_stopping": True,
                "validation_fraction": 0.1,
            },
            "use_scaled": True,
        },
        "svm": {
            "model_class": SVC,
            "param_grid": {
                "C": [0.1, 1, 5, 10],
                "kernel": ["linear", "rbf"],
                "gamma": ["scale", "auto"],
                "class_weight": [None, "balanced"],
                "probability": [True],
            },
            "fixed_params": {},
            "use_scaled": True,
        },
    }


def train_all_models(df: pd.DataFrame) -> dict:
    train_df, valid_df, test_df = split_by_year(df)

    X_train, y_train, feature_cols = prepare_xy(train_df)
    X_valid, y_valid, _ = prepare_xy(valid_df)
    X_test, y_test, _ = prepare_xy(test_df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    model_configs = get_default_param_grids(y_train)

    trained = {}

    for model_name, cfg in model_configs.items():
        train_input = X_train_scaled if cfg["use_scaled"] else X_train
        valid_input = X_valid_scaled if cfg["use_scaled"] else X_valid

        best_model, best_params, results_df = manual_tune_model(
            model_class=cfg["model_class"],
            param_grid=cfg["param_grid"],
            X_train=train_input,
            y_train=y_train,
            X_valid=valid_input,
            y_valid=y_valid,
            fixed_params=cfg["fixed_params"],
        )

        trained[model_name] = {
            "best_model": best_model,
            "best_params": best_params,
            "results": results_df,
            "use_scaled": cfg["use_scaled"],
        }

    validation_summary = pd.DataFrame([
        {
            "Model": "Logistic Regression",
            **evaluate_on_split(trained["logistic_regression"]["best_model"], X_valid_scaled, y_valid, "Validation"),
        },
        {
            "Model": "XGBoost",
            **evaluate_on_split(trained["xgboost"]["best_model"], X_valid, y_valid, "Validation"),
        },
        {
            "Model": "LightGBM",
            **evaluate_on_split(trained["lightgbm"]["best_model"], X_valid, y_valid, "Validation"),
        },
        {
            "Model": "ANN",
            **evaluate_on_split(trained["ann"]["best_model"], X_valid_scaled, y_valid, "Validation"),
        },
        {
            "Model": "SVM",
            **evaluate_on_split(trained["svm"]["best_model"], X_valid_scaled, y_valid, "Validation"),
        },
    ]).sort_values(by=["Recall", "F1", "Precision"], ascending=False).reset_index(drop=True)

    train_valid_df = df[df["Năm"] <= 2022].copy()
    X_train_valid, y_train_valid, feature_cols = prepare_xy(train_valid_df)

    scaler_final = StandardScaler()
    X_train_valid_scaled = scaler_final.fit_transform(X_train_valid)
    X_test_scaled_final = scaler_final.transform(X_test)

    final_models = {
        "Logistic Regression": (
            LogisticRegression(**trained["logistic_regression"]["best_params"]).fit(X_train_valid_scaled, y_train_valid),
            X_test_scaled_final,
        ),
        "XGBoost": (
            XGBClassifier(**trained["xgboost"]["best_params"]).fit(X_train_valid, y_train_valid),
            X_test,
        ),
        "LightGBM": (
            LGBMClassifier(**trained["lightgbm"]["best_params"]).fit(X_train_valid, y_train_valid),
            X_test,
        ),
        "ANN": (
            MLPClassifier(**trained["ann"]["best_params"]).fit(X_train_valid_scaled, y_train_valid),
            X_test_scaled_final,
        ),
        "SVM": (
            SVC(**trained["svm"]["best_params"]).fit(X_train_valid_scaled, y_train_valid),
            X_test_scaled_final,
        ),
    }

    test_summary = pd.DataFrame([
        {"Model": name, **evaluate_on_split(model, X_input, y_test, "Test")}
        for name, (model, X_input) in final_models.items()
    ]).sort_values(by=["Recall", "F1", "Precision"], ascending=False).reset_index(drop=True)

    best_model_name = test_summary.loc[0, "Model"]
    best_model_obj, best_model_X = final_models[best_model_name]

    best_y_pred, best_y_prob = get_model_predictions(best_model_obj, best_model_X)

    artifacts = {
        "feature_cols": feature_cols,
        "train_df": train_df,
        "valid_df": valid_df,
        "test_df": test_df,
        "X_test": X_test,
        "y_test": y_test,
        "validation_summary": validation_summary,
        "test_summary": test_summary,
        "best_model_name": best_model_name,
        "best_model": best_model_obj,
        "best_y_pred": best_y_pred,
        "best_y_prob": best_y_prob,
        "confusion_matrix": confusion_matrix(y_test, best_y_pred),
        "classification_report": classification_report(y_test, best_y_pred, digits=4),
        "final_models": final_models,
        "trained": trained,
        "scaler": scaler_final,
    }

    return artifacts


def tune_threshold(y_true: pd.Series, y_prob: np.ndarray, metric: str = "f1") -> tuple[pd.DataFrame, pd.Series]:
    thresholds = np.arange(0.1, 0.91, 0.05)
    rows = []

    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        rows.append({
            "threshold": th,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })

    result = pd.DataFrame(rows)
    best_row = result.sort_values(metric, ascending=False).iloc[0]
    return result, best_row


def get_curve_data(y_true: pd.Series, y_prob: np.ndarray) -> dict[str, np.ndarray | float]:
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_prob)

    return {
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "roc_auc": roc_auc_score(y_true, y_prob),
        "ap": average_precision_score(y_true, y_prob),
    }