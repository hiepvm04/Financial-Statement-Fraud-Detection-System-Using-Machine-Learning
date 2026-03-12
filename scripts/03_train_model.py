from __future__ import annotations

from pathlib import Path
import joblib
import mlflow

from src.config import (
    BEST_MODEL_FILENAME,
    FEATURES_FILENAME,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    SCALER_FILENAME,
)
from src.train import load_model_data, train_all_models


# ---- MLflow configuration ----
project_root = Path(__file__).resolve().parent.parent
mlruns_dir = project_root / "mlruns"
mlflow.set_tracking_uri(f"file:///{mlruns_dir.as_posix()}")
mlflow.set_experiment("financial-statement-fraud-detection")
# ------------------------------


def main() -> None:
    model_data_path = PROCESSED_DATA_DIR / "model_data.xlsx"

    if not model_data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file model data: {model_data_path}\n"
            "Hãy chạy scripts/02_feature_engineering.py trước."
        )

    with mlflow.start_run(run_name="train_models"):

        df = load_model_data(model_data_path)
        artifacts = train_all_models(df)

        print("=== VALIDATION SUMMARY ===")
        print(artifacts["validation_summary"].round(4))

        print("\n=== TEST SUMMARY ===")
        print(artifacts["test_summary"].round(4))

        print("\n=== BEST MODEL ===")
        print("Best model:", artifacts["best_model_name"])
        print(artifacts["classification_report"])

        best_model = artifacts["best_model"]
        scaler = artifacts["scaler"]
        feature_cols = artifacts["feature_cols"]

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        model_path = MODEL_DIR / BEST_MODEL_FILENAME
        scaler_path = MODEL_DIR / SCALER_FILENAME
        features_path = MODEL_DIR / FEATURES_FILENAME

        joblib.dump(best_model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(feature_cols, features_path)

        mlflow.log_param("best_model", artifacts["best_model_name"])
        mlflow.log_param("n_features", len(feature_cols))

        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.log_artifact(str(scaler_path), artifact_path="models")
        mlflow.log_artifact(str(features_path), artifact_path="models")

        print("\n=== ARTIFACTS SAVED ===")
        print("Model:", model_path)
        print("Scaler:", scaler_path)
        print("Features:", features_path)


if __name__ == "__main__":
    main()