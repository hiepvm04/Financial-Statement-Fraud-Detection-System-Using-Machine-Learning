from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = thư mục cha của src/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load biến môi trường từ file .env ở thư mục gốc dự án
load_dotenv(PROJECT_ROOT / ".env")

APP_NAME = os.getenv("APP_NAME", "financial-statement-fraud-detection")
APP_ENV = os.getenv("APP_ENV", "development")
DEBUG = os.getenv("DEBUG", "true").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
RAW_DATA_DIR = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
ARTIFACTS_DIR = PROJECT_ROOT / os.getenv("ARTIFACTS_DIR", "data/artifacts")

MODEL_DIR = PROJECT_ROOT / os.getenv("MODEL_DIR", "data/artifacts/models")
REPORT_DIR = PROJECT_ROOT / os.getenv("REPORT_DIR", "data/artifacts/reports")

for path in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    ARTIFACTS_DIR,
    MODEL_DIR,
    REPORT_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

VNSTOCK_API_KEY = os.getenv("VNSTOCK_API_KEY", "")
DEFAULT_SOURCE = os.getenv("DEFAULT_SOURCE", "VCI")

DEFAULT_START_YEAR = int(os.getenv("DEFAULT_START_YEAR", "2018"))
DEFAULT_END_YEAR = int(os.getenv("DEFAULT_END_YEAR", "2024"))

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TARGET_COL = os.getenv("TARGET_COL", "Fraud")
ID_COLS = ["CP", "Năm"]

TRAIN_END_YEAR = int(os.getenv("TRAIN_END_YEAR", "2021"))
VALID_YEAR = int(os.getenv("VALID_YEAR", "2022"))
TEST_YEAR = int(os.getenv("TEST_YEAR", "2023"))
TOP_K_FEATURES = int(os.getenv("TOP_K_FEATURES", "12"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
SLEEP_PER_SYMBOL = int(os.getenv("SLEEP_PER_SYMBOL", "8"))
SLEEP_PER_BATCH = int(os.getenv("SLEEP_PER_BATCH", "25"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_WORKERS = int(os.getenv("API_WORKERS", "1"))

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "financial_fraud_detection")
MLFLOW_RUN_NAME = os.getenv("MLFLOW_RUN_NAME", "baseline_run")

BEST_MODEL_FILENAME = os.getenv("BEST_MODEL_FILENAME", "best_model.joblib")
SCALER_FILENAME = os.getenv("SCALER_FILENAME", "scaler.joblib")
FEATURES_FILENAME = os.getenv("FEATURES_FILENAME", "feature_cols.joblib")