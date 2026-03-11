from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def get_confusion_matrix_df(y_true, y_pred) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(
        cm,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    )


def get_classification_report_df(y_true, y_pred) -> pd.DataFrame:
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return pd.DataFrame(report).T


def get_logistic_coefficients(model, feature_cols: list[str]) -> pd.DataFrame:
    coef_df = pd.DataFrame({
        "Feature": feature_cols,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient", ascending=False).reset_index(drop=True)
    return coef_df


def get_tree_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return importance_df


def summarize_predictions(
    df_test: pd.DataFrame,
    y_true,
    y_pred,
    y_prob,
) -> pd.DataFrame:
    result = df_test[["CP", "Năm"]].copy()
    result["Actual"] = np.array(y_true)
    result["Predicted"] = np.array(y_pred)
    result["Predicted_Prob"] = np.array(y_prob)
    return result.sort_values(["Predicted_Prob"], ascending=False).reset_index(drop=True)