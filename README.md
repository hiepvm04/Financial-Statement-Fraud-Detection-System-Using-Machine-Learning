<div align="center">

# 📊 Financial Statement Fraud Detection
**Hệ thống Phát hiện Gian lận Báo cáo Tài chính ứng dụng Học máy & MLOps**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.0+-blue.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

</div>

---

## 📖 Giới thiệu
Dự án cung cấp một giải pháp toàn diện (End-to-end Machine Learning Pipeline) nhằm **phát hiện gian lận trong báo cáo tài chính** của các công ty niêm yết trên thị trường chứng khoán Việt Nam. 

Hệ thống được thiết kế theo tiêu chuẩn **MLOps**, bao gồm từ khâu thu thập dữ liệu tự động (qua `vnstock`), làm sạch, trích xuất đặc trưng, huấn luyện nhiều mô hình học máy (XGBoost, LightGBM, ANN, SVM), theo dõi thí nghiệm bằng **MLflow**, cho đến việc triển khai mô hình (Model Serving) qua **FastAPI** và **Docker**.

---

## ✨ Tính năng nổi bật
- **⚡ Tự động hóa Dữ liệu (ETL Pipeline):** Tự động tải, làm sạch và xử lý dữ liệu tài chính từ thị trường chứng khoán Việt Nam.
- **🧠 Huấn luyện Mô hình Đa dạng:** Tự động tìm kiếm siêu tham số và đánh giá các mô hình như `Logistic Regression`, `XGBoost`, `LightGBM`, `ANN` (Neural Networks), và `SVM`.
- **📈 MLOps & Tracking:** Sẵn sàng tích hợp theo dõi quá trình huấn luyện và quản lý vòng đời mô hình bằng MLflow.
- **🚀 Triển khai API (Model Serving):** Cung cấp API dự đoán thời gian thực với độ trễ thấp sử dụng RESTful API qua `FastAPI`.
- **🐳 Dockerized:** Đóng gói toàn bộ ứng dụng và API bằng `Docker` và `docker-compose`, giúp dễ dàng triển khai trên mọi môi trường.

---

## 📁 Cấu trúc thư mục

```text
Fraud-Detection/
│
├── api/                  # FastAPI Application (Model Serving)
│   ├── main.py           # Entry point của API (endpoints)
│   ├── schemas.py        # Định nghĩa cấu trúc dữ liệu Pydantic
│   └── services.py       # Logic API dự đoán và load mô hình
│
├── data/                 # Thư mục chứa dữ liệu cục bộ (raw, processed)
├── deploy/               # Cấu hình triển khai (Dockerfile, docker-compose)
├── models/               # Nơi lưu trữ các model weights sinh ra trong khi train (.joblib)
├── notebooks/            # Jupyter Notebooks dùng để EDA và Demo
│
├── src/                  # Source code chính (Core Pipeline)
│   ├── config.py         # Cấu hình hằng số, file path
│   ├── data.py           # Tải và xử lý dữ liệu với vnstock
│   ├── preprocessing.py  # Làm sạch dữ liệu và xử lý giá trị thiếu
│   ├── features.py       # Trích xuất đặc trưng (Feature Engineering)
│   ├── train.py          # Logic huấn luyện và đánh giá nhiều mô hình
│   ├── evaluate.py       # Vẽ biểu đồ, tính toán performance metrics
│   └── pipeline.py       # Chạy full end-to-end pipeline
│
├── tests/                # Unit Tests cho dự án
├── pyproject.toml        # Quản lý dependencies, package metadata
├── .env.example          # File mẫu cho biến môi trường
└── README.md             # Project documentation (Tài liệu dự án)
```

---

## ⚙️ Cài đặt & Khởi chạy

### 1. Cài đặt trực tiếp (Local Development)

**Yêu cầu:** `Python >= 3.10`

**Bước 1:** Clone dự án và tạo môi trường ảo:
```bash
git clone https://github.com/hiepvm04/Financial-Statement-Fraud-Detection-Using-Machine-Learning.git
cd Fraud-Detection
python -m venv venv
# Chọn 1 trong 2 lệnh active dưới đây tùy theo HDH
source venv/bin/activate  # Trên Linux/Mac
venv\Scripts\activate     # Trên Windows
```

**Bước 2:** Cài đặt thư viện:
```bash
pip install -e .[all]
```

**Bước 3:** Cài đặt biến môi trường:
```bash
cp .env.example .env
```
*(Chỉnh sửa tệp `.env` nếu cần thiết).*

---

### 2. Chạy Pipeline Huấn luyện (Training)

Hệ thống có một API tiện lợi để chạy toàn bộ quy trình: *Tải dữ liệu -> Làm sạch -> Trích xuất đặc trưng -> Huấn luyện mô hình -> Lưu điểm lưu (Best Model).* 

Để chạy thử, bạn có thể tạo một script Python hoặc dùng Jupyter Notebook:
```python
from src.pipeline import run_full_pipeline

# Chạy full pipeline cho danh sách các mã cổ phiếu
artifacts = run_full_pipeline(symbols=["VCB", "FPT", "VIC", "VNM"])
```

**Theo dõi bằng MLflow:**
```bash
mlflow ui --port 5000
```
Truy cập `http://localhost:5000` trên trình duyệt để so sánh kết quả và các thông số cài đặt giữa các mô hình học máy.

---

### 3. Triển khai API (Model Serving)

Sau khi pipeline huấn luyện thành công và lưu weights vào thư mục xuất (`models/`), bạn có thể khởi động FastAPI server để thực hiện dự đoán.

**Khởi chạy trực tiếp (Local):**
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Truy cập tài liệu API tự động (Swagger UI) tại: [http://localhost:8000/docs](http://localhost:8000/docs)

**Khởi chạy bằng Docker 🐳:**
Cách tốt nhất để triển khai trên các môi trường production/testing cô lập:
```bash
cd deploy
docker-compose up --build -d
```
Hệ thống sẽ chạy song song 2 dịch vụ:
- **FastAPI Model Serving:** chạy ở port `8000`
- **MLflow Tracking Server:** chạy ở port `5000`

---

## 🛠 Thông tin API (Endpoints)

- `GET /`: Xem thông tin kiểm tra kết nối API cơ bản.
- `GET /health`: Kiểm tra trạng thái sức khỏe của API, cho biết Model/Scaler đã được nạp thành công bộ nhớ (RAM) chưa.
- `POST /predict`: Gửi các chỉ số tài chính lên và nhận lại dự đoán (nhãn Gian lận/Không gian lận & xác suất tương ứng).

---

## 💻 Tech Stack
- **Data Science / ML Pipeline:** `scikit-learn`, `XGBoost`, `LightGBM`, `Pandas`, `NumPy`
- **Data Provider:** `vnstock`
- **MLOps / Tracking:** `MLflow`
- **Backend / API Framework:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Deployment & Infra:** `Docker`, `Docker Compose`

---

## 📝 Giấy phép (License)
Dự án được phân phối dưới giấy phép mở. Vui lòng xem `pyproject.toml` để trích xuất quyền cấp phép chính thức.
