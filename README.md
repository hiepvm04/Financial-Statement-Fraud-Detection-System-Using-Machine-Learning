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
Financial-Statement-Fraud-Detection/
│
├── .github/workflows/              # Thư mục chứa CI/CD pipelines (GitHub Actions)
│   ├── train_pipeline.yml          # Tự động train lại mô hình khi có code/data mới
│   └── deploy_api.yml              # Tự động test và deploy API khi merge vào nhánh main
│
├── api/                            # Thư mục cho Model Serving (FastAPI / Flask)
│   ├── __init__.py
│   ├── main.py                     # Khởi tạo Web Server/Router (vd: POST /predict)
│   ├── schemas.py                  # Định nghĩa cấu trúc dữ liệu đầu vào/đầu ra (Pydantic models)
│   └── services.py                 # Hàm gọi model để dự đoán từ đầu vào của API
│
├── deploy/                         # [MỚI] Hạ tầng và Containerization
│   ├── Dockerfile                  # Đóng gói toàn bộ code và thư viện thành một Docker Image
│   ├── docker-compose.yml          # Chạy cục bộ cùng lúc API, MLflow Server, và Database
│   └── serve_model.sh              # Script khởi động Gunicorn/Uvicorn cho production
│
├── src/                            # Source code chính của dự án (có thể đổi tên thành fraud_detection/)
│   ├── __init__.py                 # File trống, giúp Python nhận diện thư mục này là một module
│   ├── config.py                   # (Tùy chọn) Chứa các hằng số, đường dẫn (vd: đường dẫn thư mục data/)
│   ├── data.py                     # Hàm kết nối vnstock, tải dữ liệu tài chính và lưu trữ xuống data/raw/
│   ├── preprocessing.py            # Hàm xử lý missing values, outliers, và làm sạch dữ liệu
│   ├── features.py                 # Hàm tạo đặc trưng mới (financial ratios) và feature selection
│   ├── train.py                    # Chứa Scikit-learn Pipelines, logic train và tracking với MLflow
│   └── evaluate.py                 # Hàm tính toán Metrics (Accuracy, F1-Score) và trả về dữ liệu vẽ biểu đồ
│   └── pipeline.py                 # [MỚI] Ghép nối các bước thành một ML Pipeline hoàn chỉnh
│
├── tests/                          # [BẮT BUỘC Ở MLOPS]
│   ├── test_data.py                # Kiểm thử dữ liệu đầu vào (Data Validation)
│   ├── test_model.py               # Kiểm thử model có sinh ra output hợp lệ không
│   └── test_api.py                 # Kiểm thử các endpoint của API
│
├── notebooks/                      # Thư mục chứa Jupyter Notebooks dùng để gọi hàm và trực quan hóa
│   ├── 00_data_collection.ipynb    # Gọi hàm từ src/data.py để xem trước dữ liệu tải về
│   ├── 01_preprocessing.ipynb      # Gọi hàm từ src/preprocessing.py để kiểm tra kết quả làm sạch
│   ├── 02_feature_engineering.ipynb# Gọi hàm từ src/features.py để phân tích EDA và các đặc trưng mới
│   ├── 03_model_training.ipynb     # Gọi hàm từ src/train.py để chạy thử nghiệm các mô hình
│   └── 04_model_evaluation.ipynb   # Gọi hàm từ src/evaluate.py để vẽ biểu đồ (Confusion Matrix, ROC,...)
│
├── .env.example                    # Mẫu file chứa các biến môi trường (MLFLOW_TRACKING_URI, API_KEY...)
├── requirements.txt                # Hoặc pyproject.toml / poetry.lock để quản lý dependencies chặt chẽ
└── .gitignore                      # Thêm .env, các file log, database cục bộ
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
