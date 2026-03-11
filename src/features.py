from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif

from src.config import PROCESSED_DATA_DIR, TARGET_COL, TOP_K_FEATURES

def load_processed_data(path: str | os.PathLike) -> pd.DataFrame:
    return pd.read_excel(path)


def get_candidate_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    exclude_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    exclude_cols = exclude_cols or ["CP", "Năm", target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y, feature_cols


def variance_filter(
    X: pd.DataFrame,
    threshold: float = 0.0001,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(X)

    selected = X.columns[vt.get_support()].tolist()
    removed = [col for col in X.columns if col not in selected]
    return X[selected].copy(), selected, removed


def select_by_target_correlation(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.02,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    corr_with_target = pd.concat([X, y], axis=1).corr(numeric_only=True)[y.name].drop(y.name)
    corr_with_target = corr_with_target.sort_values(key=np.abs, ascending=False)

    selected = corr_with_target[abs(corr_with_target) >= threshold].index.tolist()
    return X[selected].copy(), corr_with_target, selected


def remove_multicollinearity(
    X: pd.DataFrame,
    corr_with_target: pd.Series,
    high_corr_threshold: float = 0.80,
) -> tuple[pd.DataFrame, list[str]]:
    corr_matrix = X.corr().abs()

    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = set()

    for col in upper_triangle.columns:
        high_corr_features = upper_triangle.index[upper_triangle[col] > high_corr_threshold].tolist()
        for row_feature in high_corr_features:
            row_score = abs(corr_with_target[row_feature])
            col_score = abs(corr_with_target[col])

            if row_score >= col_score:
                to_drop.add(col)
            else:
                to_drop.add(row_feature)

    selected_after_corr = [col for col in X.columns if col not in to_drop]
    return X[selected_after_corr].copy(), sorted(to_drop)


def rank_features_by_mutual_information(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> pd.DataFrame:
    mi_scores = mutual_info_classif(X, y, random_state=random_state)

    mi_df = pd.DataFrame({
        "Feature": X.columns,
        "Mutual_Information": mi_scores,
    }).sort_values("Mutual_Information", ascending=False).reset_index(drop=True)

    return mi_df


def select_top_k_features(mi_df: pd.DataFrame, top_k: int = 12) -> list[str]:
    return mi_df.head(top_k)["Feature"].tolist()


def build_model_dataset(
    df: pd.DataFrame,
    variance_threshold: float = 0.0001,
    corr_threshold: float = 0.02,
    high_corr_threshold: float = 0.80,
    top_k: int = TOP_K_FEATURES,
    save_path: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, dict]:
    X, y, feature_cols = get_candidate_features(df)

    X_vt, vt_features, removed_low_variance = variance_filter(X, threshold=variance_threshold)

    X_corr, corr_with_target, corr_selected_features = select_by_target_correlation(
        X_vt, y, threshold=corr_threshold
    )

    X_final_candidates, removed_multicollinear = remove_multicollinearity(
        X_corr, corr_with_target, high_corr_threshold=high_corr_threshold
    )

    mi_df = rank_features_by_mutual_information(X_final_candidates, y)
    selected_features = select_top_k_features(mi_df, top_k=top_k)

    model_df = df[["CP", "Năm"] + selected_features + ["Fraud"]].copy()

    if save_path is None:
        save_path = PROCESSED_DATA_DIR / "model_data.xlsx"

    model_df.to_excel(save_path, index=False)

    info = {
        "initial_feature_count": len(feature_cols),
        "vt_features": vt_features,
        "removed_low_variance": removed_low_variance,
        "corr_with_target": corr_with_target,
        "corr_selected_features": corr_selected_features,
        "removed_multicollinear": removed_multicollinear,
        "mutual_information": mi_df,
        "selected_features": selected_features,
    }

    return model_df, info