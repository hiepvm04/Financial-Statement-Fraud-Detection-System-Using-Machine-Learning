<div align="center">

# 📊 Financial Statement Fraud Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-00a393.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.12+-blue.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-blue.svg)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*Hệ thống MLOps end-to-end phát hiện gian lận báo cáo tài chính trên thị trường chứng khoán Việt Nam.*

</div>

---

## 📌 Overview

**Financial Statement Fraud Detection System** là một giải pháp machine learning end-to-end, được thiết kế để phân tích báo cáo tài chính và phát hiện gian lận tiềm ẩn trong các công ty niêm yết trên thị trường chứng khoán Việt Nam.

Hệ thống tuân theo **quy trình MLOps hiện đại**, tự động hóa toàn bộ pipeline từ:

1. **Thu thập dữ liệu** — ETL qua `vnstock` (Bảng CĐKT, KQKD, LCTT)
2. **Tiền xử lý** — Làm sạch và chuẩn hóa dữ liệu
3. **Feature Engineering** — Tính toán >30 chỉ số tài chính
4. **Feature Selection** — Lọc đa chiều (Variance → Correlation → Mutual Information)
5. **Huấn luyện mô hình** — So sánh và tinh chỉnh 5 thuật toán ML
6. **Theo dõi thí nghiệm** — MLflow tracking (params, metrics, artifacts)
7. **Triển khai mô hình** — FastAPI REST API
8. **Containerization** — Docker + Docker Compose

---

## ✨ Key Features

### 🔄 Automated ETL Pipeline
- Thu thập dữ liệu tài chính theo lô (`BATCH_SIZE = 5`) với rate-limiting
- Tính toán tự động >30 chỉ số tài chính, bao gồm:
  - **Beneish M-Score:** DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA
  - **Accrual-based:** RSST_Accruals, Delta_Receivables, Delta_Inventory
  - **Profitability:** ROA, ROE, Net_Profit_Margin, Gross_Profit_Margin
  - **Leverage & Liquidity:** Debt_to_Assets, Working_Capital_to_Assets
  - **Governance:** Auditor_Change, Board_Independence, Issue

### 🧪 Multi-stage Feature Selection

| Bước | Phương pháp | Tham số mặc định |
|---|---|---|
| 1 | Variance Threshold | `threshold = 0.0001` |
| 2 | Target Correlation Filter | `\|r\| >= 0.02` |
| 3 | Multicollinearity Removal | `\|r\| >= 0.80` |
| 4 | Mutual Information Ranking | Top `K = 12` features |

### 🤖 Multi-Model Training & Tuning

| Mô hình | Input | Kỹ thuật tìm tham số |
|---|---|---|
| Logistic Regression | Scaled | Grid Search (C, penalty, class_weight) |
| SVM | Scaled | Grid Search (C, kernel, gamma) |
| ANN (MLP) | Scaled | Grid Search (layers, activation, alpha) |
| XGBoost | Raw | Grid Search (n_estimators, max_depth, lr) |
| LightGBM | Raw | Grid Search (n_estimators, num_leaves, lr) |

- Model tốt nhất được chọn theo thứ tự: **Recall → F1 → Precision** (trên tập test)

### 📅 Time-based Data Split (No Look-ahead Bias)

| Split | Năm |
|---|---|
| Train | ≤ 2021 |
| Validation | 2022 |
| Test | 2023 |

### 📊 MLflow Experiment Tracking
- Log params, metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
- Lưu artifacts: `best_model.joblib`, `scaler.joblib`, `feature_cols.joblib`
- MLflow UI để so sánh các experiment

### 🌐 FastAPI Serving
- `GET /health` — Kiểm tra trạng thái API và model
- `POST /predict` — Dự đoán gian lận theo thời gian thực
- Swagger UI tại `/docs`

---

## 📁 Project Structure

```text
Fraud-Detection/
│
├── api/                        # FastAPI Application
│   ├── __init__.py
│   ├── main.py                 # Endpoints: /, /health, /predict
│   ├── schemas.py              # Pydantic models (PredictionInput/Output, HealthResponse)
│   └── services.py             # Model loading & inference logic
│
├── data/
│   ├── all_symbols.xlsx        # Danh sách mã chứng khoán cần thu thập
│   ├── data_description.md     # Định nghĩa và công thức các features
│   ├── raw/                    # raw_data.xlsx (thu thập từ vnstock)
│   ├── processed/              # processed_data.xlsx, model_data.xlsx
│   └── artifacts/
│       ├── models/             # best_model.joblib, scaler.joblib, feature_cols.joblib
│       └── reports/            # Báo cáo đánh giá mô hình
│
├── deploy/
│   ├── Dockerfile              # python:3.11-slim, expose 8000
│   ├── docker-compose.yml      # fraud-api (8000) + mlflow server (5000)
│   └── serve_model.sh          # Startup script (gunicorn)
│
├── notebooks/                  # Jupyter Notebooks khám phá từng bước
│   ├── 00_data_collection.ipynb
│   ├── 01_preprocessing.ipynb
│   ├── 02_exploratory_data_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
│
├── scripts/                    # Script chạy từng bước pipeline
│   ├── run_pipeline.py         # End-to-end runner (tích hợp MLflow)
│   ├── 00_download_data.py
│   ├── 01_preprocess_data.py
│   ├── 02_feature_engineering.py
│   ├── 03_train_model.py
│   ├── 04_evaluate_model.py
│   ├── start_api.py
│   └── test_api.py
│
├── src/                        # Core ML source code
│   ├── config.py               # Cấu hình toàn cục (load từ .env)
│   ├── data.py                 # Thu thập & ETL dữ liệu (vnstock)
│   ├── preprocessing.py        # Làm sạch và chuẩn hóa
│   ├── features.py             # Feature engineering & selection
│   ├── train.py                # Huấn luyện & đánh giá mô hình
│   ├── evaluate.py             # Tiện ích đánh giá
│   └── pipeline.py             # Orchestration (data → train → save)
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
│
├── .env.example                # Mẫu biến môi trường
├── pyproject.toml              # Metadata & dependencies (PEP 517)
├── requirements.txt            # Danh sách dependency phẳng
└── README.md
```

---

## ⚙️ Installation & Getting Started

### Prerequisites
- **Python:** `>= 3.10, < 3.13`
- **Docker & Docker Compose** (tùy chọn, cho triển khai container)

---

### 1️⃣ Clone Repository

```bash
git clone https://github.com/hiepvm04/Financial-Statement-Fraud-Detection-Using-Machine-Learning.git
cd Fraud-Detection
```

### 2️⃣ Tạo Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Cài đặt Dependencies

```bash
# Cài toàn bộ (core + dev + notebooks + serving)
pip install -e .[all]

# Hoặc dùng requirements.txt
pip install -r requirements.txt
```

### 4️⃣ Cấu hình Environment Variables

```bash
cp .env.example .env
# Chỉnh sửa file .env theo nhu cầu
```

**Các biến quan trọng trong `.env`:**

| Biến | Mặc định | Mô tả |
|---|---|---|
| `VNSTOCK_API_KEY` | *(trống)* | API key cho vnstock |
| `TRAIN_END_YEAR` | `2021` | Năm cuối của tập train |
| `VALID_YEAR` | `2022` | Năm validation |
| `TEST_YEAR` | `2023` | Năm test |
| `TOP_K_FEATURES` | `12` | Số features được chọn |
| `MLFLOW_TRACKING_URI` | `http://127.0.0.1:5000` | URI MLflow server |
| `MLFLOW_EXPERIMENT_NAME` | `financial_fraud_detection` | Tên experiment |
| `BATCH_SIZE` | `5` | Số mã/lần thu thập |
| `SLEEP_PER_SYMBOL` | `8` | Delay giữa các mã (giây) |
| `SLEEP_PER_BATCH` | `25` | Delay giữa các batch (giây) |
| `DEFAULT_SOURCE` | `VCI` | Nguồn dữ liệu vnstock |

---

## 🚀 Running the Pipeline

### Chạy toàn bộ Pipeline (End-to-End)

```bash
python -m scripts.run_pipeline
```

Pipeline sẽ thực hiện tuần tự:
1. Nạp 500 mã từ `data/all_symbols.xlsx`
2. Thu thập báo cáo tài chính qua `vnstock`
3. Tiền xử lý và làm sạch dữ liệu
4. Feature engineering & selection (top 12 features)
5. Huấn luyện và tinh chỉnh 5 mô hình ML
6. Lưu model tốt nhất vào `data/artifacts/models/`
7. Log toàn bộ kết quả lên **MLflow**

### Chạy từng bước riêng lẻ

```bash
python -m scripts.00_download_data       # Thu thập dữ liệu
python -m scripts.01_preprocess_data     # Tiền xử lý
python -m scripts.02_feature_engineering # Feature selection
python -m scripts.03_train_model         # Huấn luyện
python -m scripts.04_evaluate_model      # Đánh giá
```

### Xem MLflow UI

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

Truy cập: [http://localhost:5000](http://localhost:5000)

---

## 🌐 API Deployment

### Chạy API Locally

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

| URL | Mô tả |
|---|---|
| http://localhost:8000 | API Root |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/health | Health Check |

### Triển khai với Docker Compose

```bash
cd deploy
docker-compose up --build -d
```

| Service | URL |
|---|---|
| Fraud Detection API | http://localhost:8000 |
| MLflow Tracking UI | http://localhost:5000 |

```bash
# Dừng tất cả services
docker-compose down
```

---

## 📡 API Reference

### `GET /health` — Health Check

```bash
curl -X GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_count": 12,
  "model_name": "XGBClassifier"
}
```

---

### `POST /predict` — Dự đoán Gian lận

Request body gồm **12 features Beneish M-Score / Accrual** được chọn trong quá trình training:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "DSRI": 1.15,
    "GMI": 0.95,
    "AQI": 1.05,
    "SGI": 1.20,
    "DEPI": 0.98,
    "SGAI": 1.02,
    "LVGI": 1.10,
    "TATA": 0.03,
    "RSST_Accruals": 0.05,
    "Delta_Receivables": 0.02,
    "Delta_Inventory": 0.01,
    "Delta_Cash_Sales": -0.01
  }'
```

**Request Fields (`PredictionInput`):**

| Field | Mô tả |
|---|---|
| `DSRI` | Days Sales in Receivables Index |
| `GMI` | Gross Margin Index |
| `AQI` | Asset Quality Index |
| `SGI` | Sales Growth Index |
| `DEPI` | Depreciation Index |
| `SGAI` | SG&A Expense Index |
| `LVGI` | Leverage Index |
| `TATA` | Total Accruals to Total Assets |
| `RSST_Accruals` | RSST Accrual Quality |
| `Delta_Receivables` | Thay đổi Khoản phải thu / Tổng tài sản |
| `Delta_Inventory` | Thay đổi Hàng tồn kho / Tổng tài sản |
| `Delta_Cash_Sales` | Thay đổi Doanh thu tiền mặt |

**Response (`PredictionOutput`):**
```json
{
  "label": "Fraud",
  "fraud_prediction": 1,
  "fraud_probability": 0.8923,
  "threshold": 0.5,
  "model_name": "XGBClassifier"
}
```

---

## 🧪 Running Tests

```bash
# Chạy toàn bộ tests
pytest

# Kèm báo cáo coverage
pytest --cov=src --cov=api --cov-report=term-missing
```

---

## 📦 Tech Stack

| Nhóm | Thư viện |
|---|---|
| **Data Processing** | `pandas >= 2.0`, `numpy >= 1.24`, `scipy >= 1.10` |
| **Machine Learning** | `scikit-learn >= 1.3`, `xgboost >= 2.0`, `lightgbm >= 4.0` |
| **Data Source** | `vnstock >= 3.0` |
| **Experiment Tracking** | `mlflow >= 2.12`, `joblib >= 1.3` |
| **API & Serving** | `fastapi >= 0.110`, `uvicorn >= 0.29`, `pydantic >= 2.6`, `gunicorn >= 21.2` |
| **Infrastructure** | `Docker`, `Docker Compose` |
| **Dev & Quality** | `pytest`, `black`, `ruff`, `mypy`, `pre-commit` |
| **Notebooks** | `jupyter`, `matplotlib`, `seaborn`, `openpyxl` |

---

## 📬 Contact

Đây là dự án cá nhân cho mục đích học tập và nghiên cứu. Nếu bạn thấy hữu ích hoặc có góp ý, hãy mở Issue hoặc liên hệ:

- **Tác giả:** Vũ Mạnh Hiệp
- **Email:** hiepvm04@gmail.com
- **GitHub:** [github.com/hiepvm04](https://github.com/hiepvm04)
