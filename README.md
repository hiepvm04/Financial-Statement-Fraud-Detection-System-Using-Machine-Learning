<div align="center">

# 📊 Hệ thống Phát hiện Gian lận Báo cáo Tài chính
**(Financial Statement Fraud Detection System)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.2+-blue.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-blue?logo=lightgbm)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

*Giải pháp End-to-End Machine Learning (MLOps) tự động phát hiện gian lận tài chính tại thị trường chứng khoán Việt Nam.*
</div>

---

## 📖 Giới thiệu
Dự án **Financial Statement Fraud Detection** cung cấp một giải pháp toàn diện và tự động nhằm phân tích, đánh giá và dự đoán rủi ro gian lận trong báo cáo tài chính của các công ty niêm yết trên thị trường chứng khoán Việt Nam. 

Hệ thống được thiết kế theo tiêu chuẩn công nghiệp (MLOps), tự động hóa hoàn toàn từ khâu thu thập dữ liệu (ETL qua `vnstock`), làm sạch, trích xuất đặc trưng (Feature Engineering), huấn luyện các mô hình Machine Learning mạnh mẽ, cho đến việc theo dõi thí nghiệm (MLflow) và đóng gói triển khai (Docker + FastAPI).

---

## ✨ Điểm nổi bật (Key Features)

- **🔄 ETL Pipeline Tự động:** Tích hợp trực tiếp dữ liệu từ `vnstock`, xử lý báo cáo KQKD, Bảng cân đối kế toán, Lưu chuyển tiền tệ. Tự động tính toán các đặc trưng tài chính (Tài sản mềm, DSRI, AQI, Điểm số Accruals...).
- **🧠 Kỹ thuật Trích xuất Đặc trưng Nâng cao:** Sử dụng `VarianceThreshold`, loại bỏ đa cộng tuyến (Multicollinearity), và chọn lọc đặc trưng tối ưu (Feature Selection) thông qua `Mutual Information`.
- **🚀 Huấn luyện Đa Mô Hình (Multi-Model Training):** Tự động tinh chỉnh siêu tham số (Hyperparameter Tuning) và so sánh nhiều mô hình: `Logistic Regression`, `XGBoost`, `LightGBM`, `ANN` (MLPClassifier), và `SVM`.
- **📈 MLflow Tracking:** Quản lý vòng đời mô hình chuyên nghiệp. Lưu trữ chi tiết các chỉ số (Accuracy, ROC-AUC, F1-Score, Precision-Recall) và tự động chọn ra Best Model cho từng lần huấn luyện.
- **⚡ Model Serving với FastAPI:** Cung cấp RESTful API với độ trễ thấp, phục vụ dự đoán thời gian thực.
- **🐳 Sẵn sàng Triển khai (Dockerized):** Đóng gói toàn bộ cấu phần API và MLflow Server thành các container độc lập qua `docker-compose`.

---

## 📁 Cấu trúc thư mục (Project Structure)

```text
Fraud-Detection/
│
├── api/                  # FastAPI Application (Model Serving)
│   ├── main.py           # Khai báo cấu hình FastAPI, định tuyến (endpoints)
│   ├── schemas.py        # Pydantic models để validate dữ liệu I/O
│   └── services.py       # Services script (load memory, model prediction)
│
├── data/                 # Thư mục chứa dữ liệu local (Raw & Processed)
├── deploy/               # Cấu hình triển khai hạ tầng
│   ├── Dockerfile        # Cấu trúc image cho FastAPI
│   └── docker-compose.yml# Chạy đồng thời hệ thống FastAPI & MLflow
│
├── models/               # Nơi lưu Artifacts sinh ra (Best Model, Scaler, Features)
├── notebooks/            # Jupyter Notebooks minh họa từng bước xử lý dữ liệu, EDA
│
├── src/                  # Mã nguồn lõi của ML Pipeline
│   ├── config.py         # Cấu hình hằng số (paths, parameters)
│   ├── data.py           # Logic ETL - Kết nối vnstock, sinh chỉ số tài chính, định dạnh nhãn (Fraud)
│   ├── preprocessing.py  # Xử lý missing values, làm sạch (Data Cleaning)
│   ├── features.py       # Code xử lý Feature Selection & Engineering
│   ├── train.py          # Script grid-search, train, evaluate các model
│   ├── evaluate.py       # Metrics & visualize kết quả đánh giá (ROC, PR Curves)
│   └── pipeline.py       # Khởi tạo chu trình full End-to-End
│
├── tests/                # System & Unit test files
├── pyproject.toml        # Quản lý thư viện hệ thống (dependencies)
├── .env.example          # Thông tin cấu hình môi trường bảo mật
└── README.md             # Project documentation (Tài liệu dự án)
```

---

## ⚙️ Hướng dẫn Cài đặt & Khởi chạy (Getting Started)

### 📋 Yêu cầu hệ thống (Prerequisites)
- **OS:** Windows, macOS, Linux.
- **Python:** `>= 3.10`
- **Docker & Docker Compose** (Nếu dùng cấu hình production)

### 1. Cài đặt trực tiếp (Local Setup)

```bash
# 1. Clone dự án
git clone https://github.com/hiepvm04/Financial-Statement-Fraud-Detection-Using-Machine-Learning.git
cd Fraud-Detection

# 2. Tạo Virtual Environment và kích hoạt
python -m venv venv
source venv/bin/activate       # Trên macOS/Linux
venv\Scripts\activate          # Trên Windows

# 3. Cài đặt toàn bộ dependencies (bao gồm API, Notebook, Test libs)
pip install -e .[all]

# 4. Cấu hình biến môi trường
cp .env.example .env  # Cập nhật VNSTOCK_API_KEY nếu cần thiết trong .env
```

### 2. Chạy Pipeline Huấn luyện (Training Pipeline)

Dự án cung cấp luồng chạy tích hợp (từ Tải dữ liệu -> Làm sạch -> Trích xuất đặc trưng -> Train -> Lưu Model).

Bạn có thể chạy tự động toàn bộ tiến trình thông qua module `pipeline.py`:
```python
# Tạo tệp lệnh run.py tại thư mục root
from src.pipeline import run_full_pipeline

# Pipeline sẽ kết nối Vnstock, down danh sách cổ phiếu cần phân tích 
# và chạy quy trình huấn luyện đầy đủ
run_full_pipeline(symbols=["VCB", "FPT", "VIC", "VNM", "SSI"])
```
*Ghi chú: Mô hình tốt nhất (cùng với Scaler và danh sách các features được chọn) sẽ tự động lưu vào thư mục `models/`.*

### 🔍 Quản lý mô hình bằng MLflow
Để kiểm tra biểu đồ độ chính xác và lịch sử thông số của các mô hình đã được huấn luyện:
```bash
mlflow ui --port 5000
```
Truy cập: [http://localhost:5000](http://localhost:5000)

---

## 🚀 Triển khai API (Model Serving)

Sau khi Pipeline Training hoàn tất và Best Model đã được lưu, API Serve sẵn sàng phục vụ.

### 💻 Khởi chạy Local
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Live API Endpoint:** [http://localhost:8000](http://localhost:8000)
- **Interactive API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

### 🐳 Khởi chạy Production bằng Docker

Môi trường Docker sẽ chạy đồng thời Web Server cho API và Tracking Server cho MLflow.
```bash
cd deploy
docker-compose up --build -d
```
API sẽ lắng nghe ở port `:8000` và MLflow ở port `:5000`. Để dừng toàn bộ hệ thống, sử dụng `docker-compose down`.

---

## 🛠 Tài liệu API (API Reference)

### 1. Kiểm tra trạng thái hệ thống: `GET /health`
```bash
curl -X 'GET' 'http://localhost:8000/health' -H 'accept: application/json'
```
*Đầu ra mong đợi:*
```json
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_count": 12,
  "model_name": "XGBClassifier"
}
```

### 2. Gửi yêu cầu dự đoán: `POST /predict`
Body request yêu cầu chứa chính xác các `features` đã được lựa chọn ở bước huấn luyện.

```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "DSRI": 1.15,
  "GMI": 0.95,
  "AQI": 1.05,
  "SGI": 1.2,
  "TATA": 0.03,
  "Firm_Size": 15.5
}'
```
*(Ghi chú: Số lượng features đầu vào có thể thay đổi tùy thuộc vào cấu hình Pipeline tự động lựa chọn đặc trưng)*

*Đầu ra mong đợi:*
```json
{
  "label": "Fraud",
  "fraud_prediction": 1,
  "fraud_probability": 0.892345,
  "threshold": 0.5,
  "model_name": "XGBClassifier"
}
```

---

## 💻 Công nghệ Cốt lõi (Tech Stack)
- **Machine Learning Data Processing:** `Pandas`, `NumPy`, `Scikit-Learn`
- **Core ML Algorithms:** `XGBoost`, `LightGBM`, `Scikit-Learn (SVM, Logsitic Regression, ANN)`
- **Data Source API:** `vnstock` (Truy xuất dữ liệu thị trường Việt Nam)
- **Model Monitoring:** `MLflow` (Experiment tracking & Artifact storage)
- **Backend & Web API:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Infrastructure:** `Docker`, `Docker-Compose`

---

## 📝 Giấy phép (License)
Dự án được phân phối dưới giấy phép MIT. Xem tệp `pyproject.toml` để biết thêm chi tiết.
