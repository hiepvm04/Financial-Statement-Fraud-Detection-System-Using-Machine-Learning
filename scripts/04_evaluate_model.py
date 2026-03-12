from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns

from src.config import PROCESSED_DATA_DIR
from src.evaluate import (
    get_classification_report_df,
    get_confusion_matrix_df,
    get_logistic_coefficients,
    get_tree_feature_importance,
    summarize_predictions,
)
from src.train import get_model_predictions, load_model_data, train_all_models

project_root = Path(__file__).resolve().parent.parent
mlruns_dir = project_root / "mlruns"
mlflow.set_tracking_uri(f"file:///{mlruns_dir.as_posix()}")
mlflow.set_experiment("financial-statement-fraud-detection")

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def main() -> None:
    model_data_path = PROCESSED_DATA_DIR / "model_data.xlsx"

    if not model_data_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file model data: {model_data_path}\n"
            "Hãy chạy scripts/02_feature_engineering.py trước."
        )
    with mlflow.start_run(run_name="model_evaluation"):
        df = load_model_data(model_data_path)
        artifacts = train_all_models(df)

        best_model_name = artifacts["best_model_name"]
        best_model = artifacts["best_model"]
        y_test = artifacts["y_test"]
        feature_cols = artifacts["feature_cols"]

        final_models = artifacts["final_models"]
        best_model_obj, best_X_test = final_models[best_model_name]

        best_y_pred, best_y_prob = get_model_predictions(best_model_obj, best_X_test)

        print("=== TEST SUMMARY ===")
        print(artifacts["test_summary"].round(4))

        print("\n=== CLASSIFICATION REPORT ===")
        print(artifacts["classification_report"])

        mlflow.log_param("best_model_name", best_model_name)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("n_test_samples", len(y_test))

        test_summary = artifacts["test_summary"]
        if hasattr(test_summary, "loc") and best_model_name in test_summary.index:
            row = test_summary.loc[best_model_name]
            for metric_name, metric_value in row.items():
                if isinstance(metric_value, (int, float)):
                    mlflow.log_metric(f"test_{metric_name}", float(metric_value))

        cm_df = get_confusion_matrix_df(y_test, best_y_pred)
        print("\n=== CONFUSION MATRIX ===")
        print(cm_df)

        report_df = get_classification_report_df(y_test, best_y_pred)
        print("\n=== CLASSIFICATION REPORT DF ===")
        print(report_df)

        prediction_df = summarize_predictions(
            df_test=artifacts["test_df"],
            y_true=y_test,
            y_pred=best_y_pred,
            y_prob=best_y_prob,
        )
        print("\n=== TOP PREDICTIONS ===")
        print(prediction_df.head(20))

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Confusion matrix plot
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {best_model_name}")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()

            cm_plot_path = tmpdir_path / "confusion_matrix.png"
            plt.savefig(cm_plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            mlflow.log_artifact(str(cm_plot_path), artifact_path="plots")

            # Save report tables
            cm_csv_path = tmpdir_path / "confusion_matrix.csv"
            report_csv_path = tmpdir_path / "classification_report.csv"
            preds_csv_path = tmpdir_path / "top_predictions.csv"

            cm_df.to_csv(cm_csv_path, index=True)
            report_df.to_csv(report_csv_path, index=True)
            prediction_df.to_csv(preds_csv_path, index=False)

            mlflow.log_artifact(str(cm_csv_path), artifact_path="tables")
            mlflow.log_artifact(str(report_csv_path), artifact_path="tables")
            mlflow.log_artifact(str(preds_csv_path), artifact_path="tables")

            if best_model_name == "Logistic Regression":
                coef_df = get_logistic_coefficients(best_model, feature_cols)
                print("\n=== LOGISTIC COEFFICIENTS ===")
                print(coef_df)

                coef_csv_path = tmpdir_path / "logistic_coefficients.csv"
                coef_df.to_csv(coef_csv_path, index=False)
                mlflow.log_artifact(str(coef_csv_path), artifact_path="tables")

                plt.figure(figsize=(10, 6))
                sns.barplot(data=coef_df, x="Coefficient", y="Feature")
                plt.title("Logistic Regression Coefficients")
                plt.tight_layout()

                coef_plot_path = tmpdir_path / "logistic_coefficients.png"
                plt.savefig(coef_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(str(coef_plot_path), artifact_path="plots")

            if best_model_name in ["XGBoost", "LightGBM"]:
                importance_df = get_tree_feature_importance(best_model, feature_cols)
                print("\n=== FEATURE IMPORTANCE ===")
                print(importance_df)

                importance_csv_path = tmpdir_path / "feature_importance.csv"
                importance_df.to_csv(importance_csv_path, index=False)
                mlflow.log_artifact(str(importance_csv_path), artifact_path="tables")

                plt.figure(figsize=(10, 6))
                sns.barplot(data=importance_df, x="Importance", y="Feature")
                plt.title(f"Feature Importance - {best_model_name}")
                plt.tight_layout()

                importance_plot_path = tmpdir_path / "feature_importance.png"
                plt.savefig(importance_plot_path, dpi=300, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(str(importance_plot_path), artifact_path="plots")


if __name__ == "__main__":
    main()