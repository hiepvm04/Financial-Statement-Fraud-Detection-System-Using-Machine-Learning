from __future__ import annotations

import numpy as np
import pandas as pd

from src.preprocessing import fill_missing_values, clip_outliers
from src.features import (
    get_candidate_features,
    variance_filter,
    select_by_target_correlation,
    remove_multicollinearity,
    rank_features_by_mutual_information,
    select_top_k_features,
)


def make_sample_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CP": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
            "Năm": [2020, 2021, 2020, 2021, 2020, 2021],
            "DSRI": [1.1, np.nan, 0.9, 1.0, 1.2, 1.15],
            "GMI": [0.95, 1.05, np.nan, 1.10, 0.98, 1.00],
            "AQI": [1.02, 1.04, 1.01, np.nan, 1.08, 1.06],
            "SGI": [1.10, 1.20, 0.95, 1.05, 1.15, 1.18],
            "TATA": [0.01, 0.03, -0.02, 0.00, 0.05, 0.04],
            "Firm_Size": [20.0, 20.2, 19.5, 19.7, 21.0, 21.1],
            "Fraud": [0, 1, 0, 0, 1, 1],
        }
    )


def test_fill_missing_values_removes_missing_in_numeric_columns():
    df = make_sample_dataset()

    result = fill_missing_values(df, group_col="CP", exclude_cols=["Fraud"])

    numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Fraud"]

    assert result[numeric_cols].isna().sum().sum() == 0


def test_clip_outliers_keeps_shape_and_columns():
    df = make_sample_dataset()
    df = fill_missing_values(df, group_col="CP", exclude_cols=["Fraud"])

    result = clip_outliers(df, lower_q=0.01, upper_q=0.99, exclude_cols=["Fraud"])

    assert result.shape == df.shape
    assert list(result.columns) == list(df.columns)


def test_get_candidate_features_excludes_id_and_target_columns():
    df = make_sample_dataset()

    X, y, feature_cols = get_candidate_features(df, target_col="Fraud")

    assert "CP" not in feature_cols
    assert "Năm" not in feature_cols
    assert "Fraud" not in feature_cols
    assert len(X.columns) == len(feature_cols)
    assert y.name == "Fraud"


def test_variance_filter_removes_constant_feature():
    df = make_sample_dataset()
    df["constant_feature"] = 1.0

    X, _, _ = get_candidate_features(df, target_col="Fraud")
    X_vt, selected, removed = variance_filter(X, threshold=0.0001)

    assert "constant_feature" in removed
    assert "constant_feature" not in selected
    assert "constant_feature" not in X_vt.columns


def test_correlation_and_multicollinearity_pipeline_runs():
    df = make_sample_dataset()
    df = fill_missing_values(df, group_col="CP", exclude_cols=["Fraud"])

    X, y, _ = get_candidate_features(df, target_col="Fraud")
    X_vt, _, _ = variance_filter(X, threshold=0.0)

    X_corr, corr_with_target, selected = select_by_target_correlation(
        X_vt, y, threshold=0.0
    )

    X_final, removed_multicollinear = remove_multicollinearity(
        X_corr, corr_with_target, high_corr_threshold=0.95
    )

    assert isinstance(selected, list)
    assert isinstance(removed_multicollinear, list)
    assert isinstance(X_final, pd.DataFrame)
    assert X_final.shape[0] == df.shape[0]


def test_mutual_information_ranking_and_top_k():
    df = make_sample_dataset()
    df = fill_missing_values(df, group_col="CP", exclude_cols=["Fraud"])

    X, y, _ = get_candidate_features(df, target_col="Fraud")
    X_vt, _, _ = variance_filter(X, threshold=0.0)

    mi_df = rank_features_by_mutual_information(X_vt, y, random_state=42)
    selected_features = select_top_k_features(mi_df, top_k=3)

    assert "Feature" in mi_df.columns
    assert "Mutual_Information" in mi_df.columns
    assert len(selected_features) == 3
    assert all(feature in X_vt.columns for feature in selected_features)