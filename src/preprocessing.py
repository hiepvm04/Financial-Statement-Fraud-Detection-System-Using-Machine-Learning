from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DATA_DIR, TARGET_COL


def load_raw_data(path: str | os.PathLike) -> pd.DataFrame:
    return pd.read_excel(path)


def summarize_data(df: pd.DataFrame) -> dict[str, pd.DataFrame | int | tuple[int, int]]:
    missing = df.isnull().sum().sort_values(ascending=False)
    missing_percent = df.isnull().mean().mul(100)

    missing_df = pd.concat([missing, missing_percent], axis=1)
    missing_df.columns = ["Missing_Count", "Missing_%"]

    numeric_summary = df.describe(include=[np.number]).T if not df.empty else pd.DataFrame()

    summary = {
        "shape": df.shape,
        "columns": pd.DataFrame({"columns": df.columns}),
        "missing": missing_df,
        "dtypes": pd.DataFrame({"dtype": df.dtypes.astype(str)}),
        "numeric_summary": numeric_summary,
        "duplicate_rows": int(df.duplicated().sum()),
    }

    if {"CP", "Năm"}.issubset(df.columns):
        summary["duplicate_cp_year"] = int(df.duplicated(subset=["CP", "Năm"]).sum())

    return summary


def fill_missing_values(
    df: pd.DataFrame,
    group_col: str = "CP",
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fill missing:
    1) mean theo từng công ty
    2) median toàn dataset
    """
    result = df.copy()
    exclude_cols = exclude_cols or [TARGET_COL]

    numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    if group_col in result.columns:
        result[numeric_cols] = result.groupby(group_col)[numeric_cols].transform(
            lambda x: x.fillna(x.mean())
        )

    result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].median())
    return result


def clip_outliers(
    df: pd.DataFrame,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
    exclude_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Clip numeric columns theo quantile.
    """
    result = df.copy()
    exclude_cols = exclude_cols or [TARGET_COL]

    numeric_cols = result.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    result[numeric_cols] = result[numeric_cols].clip(
        lower=result[numeric_cols].quantile(lower_q),
        upper=result[numeric_cols].quantile(upper_q),
        axis=1,
    )
    return result


def preprocess_dataset(
    df: pd.DataFrame,
    save_path: str | os.PathLike | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    """
    result = fill_missing_values(df, group_col="CP", exclude_cols=[TARGET_COL])
    result = clip_outliers(result, lower_q=0.01, upper_q=0.99, exclude_cols=[TARGET_COL])

    if save_path is None:
        save_path = PROCESSED_DATA_DIR / "processed_data.xlsx"

    result.to_excel(save_path, index=False)
    print(f"Processed data saved to: {save_path}")
    return result