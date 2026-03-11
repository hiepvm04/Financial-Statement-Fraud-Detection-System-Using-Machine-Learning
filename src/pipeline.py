from __future__ import annotations

import os
from typing import Iterable

import joblib
import pandas as pd

from src.config import (
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    BEST_MODEL_FILENAME,
    SCALER_FILENAME,
    FEATURES_FILENAME,
)
from src.data import collect_financial_dataset, set_vnstock_api_key
from src.features import build_model_dataset
from src.preprocessing import preprocess_dataset
from src.train import train_all_models


def run_data_pipeline(
    symbols: Iterable[str],
    api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chạy từ raw -> processed -> model_data.
    """
    set_vnstock_api_key(api_key)

    raw_df, failed_symbols = collect_financial_dataset(
        symbols=symbols,
        save_path=RAW_DATA_DIR / "raw_data.xlsx",
    )

    processed_df = preprocess_dataset(
        raw_df,
        save_path=PROCESSED_DATA_DIR / "processed_data.xlsx",
    )

    model_df, feature_info = build_model_dataset(
        processed_df,
        save_path=PROCESSED_DATA_DIR / "model_data.xlsx",
    )

    print("Pipeline dữ liệu hoàn tất.")
    print("Failed symbols:", failed_symbols)
    print("Selected features:", feature_info["selected_features"])

    return raw_df, processed_df, model_df


def run_training_pipeline(model_data_path: str | os.PathLike | None = None) -> dict:
    """
    Train tất cả model và lưu best model.
    """
    if model_data_path is None:
        model_data_path = PROCESSED_DATA_DIR / "model_data.xlsx"

    df = pd.read_excel(model_data_path)
    artifacts = train_all_models(df)

    best_model_name = artifacts["best_model_name"]
    best_model = artifacts["best_model"]
    scaler = artifacts["scaler"]
    feature_cols = artifacts["feature_cols"]

    joblib.dump(best_model, MODEL_DIR / BEST_MODEL_FILENAME)
    joblib.dump(scaler, MODEL_DIR / SCALER_FILENAME)
    joblib.dump(feature_cols, MODEL_DIR / FEATURES_FILENAME)

    print(f"Best model saved: {best_model_name}")
    print(f"Validation summary:\n{artifacts['validation_summary']}")
    print(f"Test summary:\n{artifacts['test_summary']}")

    return artifacts


def run_full_pipeline(
    symbols: Iterable[str],
    api_key: str | None = None,
) -> dict:
    """
    Chạy full end-to-end pipeline:
    data collection -> preprocessing -> feature selection -> training
    """
    run_data_pipeline(symbols=symbols, api_key=api_key)
    artifacts = run_training_pipeline()
    return artifacts