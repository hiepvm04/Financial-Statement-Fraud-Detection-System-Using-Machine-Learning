from __future__ import annotations

from pathlib import Path

import mlflow

from src.config import (
    BEST_MODEL_FILENAME,
    FEATURES_FILENAME,
    MODEL_DIR,
    RAW_DATA_DIR,
    SCALER_FILENAME,
    VNSTOCK_API_KEY,
)
from src.data import load_symbols_from_excel
from src.pipeline import run_full_pipeline


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    symbols_file = project_root / "data/all_symbols.xlsx"

    if not symbols_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {symbols_file}")

    symbols = load_symbols_from_excel(
        excel_path=symbols_file,
        top_n=500,
    )

    mlflow.set_experiment("financial-statement-fraud-detection")

    with mlflow.start_run(run_name="full_pipeline_run"):
        mlflow.log_param("symbols_file", str(symbols_file))
        mlflow.log_param("top_n", 500)
        mlflow.log_param("n_symbols", len(symbols))

        print("=== RUN FULL PIPELINE ===")
        print("Số mã sẽ xử lý:", len(symbols))

        artifacts = run_full_pipeline(
            symbols=symbols,
            api_key=VNSTOCK_API_KEY,
        )

        print("\n=== PIPELINE COMPLETED ===")
        print("Best model:", artifacts["best_model_name"])
        print("\nValidation summary:")
        print(artifacts["validation_summary"].round(4))
        print("\nTest summary:")
        print(artifacts["test_summary"].round(4))

        best_model_name = artifacts["best_model_name"]
        validation_summary = artifacts["validation_summary"]
        test_summary = artifacts["test_summary"]

        mlflow.log_param("best_model_name", best_model_name)

        if hasattr(validation_summary, "loc") and best_model_name in validation_summary.index:
            val_row = validation_summary.loc[best_model_name]
            for metric_name, metric_value in val_row.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"val_{metric_name}", float(metric_value))

        if hasattr(test_summary, "loc") and best_model_name in test_summary.index:
            test_row = test_summary.loc[best_model_name]
            for metric_name, metric_value in test_row.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", float(metric_value))

        model_path = MODEL_DIR / BEST_MODEL_FILENAME
        scaler_path = MODEL_DIR / SCALER_FILENAME
        features_path = MODEL_DIR / FEATURES_FILENAME

        if model_path.exists():
            mlflow.log_artifact(str(model_path), artifact_path="models")
        if scaler_path.exists():
            mlflow.log_artifact(str(scaler_path), artifact_path="models")
        if features_path.exists():
            mlflow.log_artifact(str(features_path), artifact_path="models")


if __name__ == "__main__":
    main()