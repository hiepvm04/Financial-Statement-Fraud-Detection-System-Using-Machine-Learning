from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.config import (
    PROJECT_ROOT,
    MODEL_DIR,
    BEST_MODEL_FILENAME,
    SCALER_FILENAME,
    FEATURES_FILENAME,
)

load_dotenv(PROJECT_ROOT / ".env")


PREDICTION_THRESHOLD = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))


class ModelService:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.scaler: Any | None = None
        self.feature_cols: list[str] = []
        self.model_name: str = "unknown"

    def load_artifacts(self) -> None:
        model_path = MODEL_DIR / BEST_MODEL_FILENAME
        scaler_path = MODEL_DIR / SCALER_FILENAME
        features_path = MODEL_DIR / FEATURES_FILENAME

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")

        if not features_path.exists():
            raise FileNotFoundError(f"Feature file not found: {features_path}")

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.feature_cols = joblib.load(features_path)

        self.model_name = self.model.__class__.__name__

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.scaler is not None and len(self.feature_cols) > 0

    def _build_dataframe(self, payload: dict[str, float]) -> pd.DataFrame:
        df = pd.DataFrame([payload])

        missing_cols = [col for col in self.feature_cols if col not in df.columns]
        extra_cols = [col for col in df.columns if col not in self.feature_cols]

        if missing_cols:
            raise ValueError(f"Missing required features: {missing_cols}")

        if extra_cols:
            df = df.drop(columns=extra_cols)

        df = df[self.feature_cols]
        return df

    def _transform(self, X: pd.DataFrame) -> Any:
        scaled_model_names = {
            "LogisticRegression",
            "MLPClassifier",
            "SVC",
        }

        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        if self.model.__class__.__name__ in scaled_model_names:
            return self.scaler.transform(X)

        return X

    def predict_proba(self, payload: dict[str, float]) -> float:
        if not self.is_loaded:
            raise RuntimeError("Artifacts are not loaded.")

        X = self._build_dataframe(payload)
        X_input = self._transform(X)

        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X_input)[:, 1][0])
        else:
            scores = self.model.decision_function(X_input)
            prob = float(1 / (1 + np.exp(-scores[0])))

        return prob

    def predict(self, payload: dict[str, float], threshold: float = PREDICTION_THRESHOLD) -> dict[str, Any]:
        fraud_probability = self.predict_proba(payload)
        fraud_prediction = int(fraud_probability >= threshold)
        label = "Fraud" if fraud_prediction == 1 else "Non-Fraud"

        return {
            "label": label,
            "fraud_prediction": fraud_prediction,
            "fraud_probability": round(fraud_probability, 6),
            "threshold": threshold,
            "model_name": self.model_name,
        }


model_service = ModelService()