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

## ⚙️ Hướng dẫn Cài đặt & Khởi chạy chi tiết

Dưới đây là các bước chi tiết để thiết lập môi trường, chạy pipeline huấn luyện và khởi tạo API dự đoán.

### 📋 Yêu cầu hệ thống (Prerequisites)
- **Hệ điều hành:** Windows, macOS, hoặc Linux
- **Python:** Phiên bản `>= 3.10` (Khuyến nghị 3.10 hoặc 3.11)
- **Git:** Để clone mã nguồn dự án
- **Docker & Docker Compose:** (Tùy chọn) - Dành cho việc triển khai vào môi trường ảo hóa.

---

### 1. Cài đặt trực tiếp (Local Development)

**Bước 1: Tải mã nguồn**
Mở Terminal (hoặc PowerShell trên Windows) và chạy lệnh:
```bash
git clone https://github.com/hiepvm04/Financial-Statement-Fraud-Detection-Using-Machine-Learning.git
cd Fraud-Detection
```

**Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)**
Việc sử dụng môi trường ảo giúp tránh xung đột các thư viện Python trên máy tính của bạn:
```bash
# Tạo môi trường ảo có tên "venv"
python -m venv venv

# Kích hoạt trên Windows (PowerShell/CMD):
venv\Scripts\activate

# Kích hoạt trên Linux/macOS:
source venv/bin/activate
```
*(Dấu hiệu thành công: Đầu dòng lệnh của bạn sẽ xuất hiện chữ `(venv)`).*

**Bước 3: Cài đặt các thư viện cần thiết**
Dự án được cấu hình bằng `pyproject.toml`. Lệnh dưới đây sẽ cài đặt toàn bộ core pipeline, công cụ dev, notebooks và web server:
```bash
pip install -e .[all]
```

**Bước 4: Cấu hình biến môi trường**
Dự án sử dụng file `.env` để bảo mật các cấu hình (chẳng hạn api key, port). 
Tạo file `.env` từ file mẫu:
- Trên **Windows (PowerShell)**: `copy .env.example .env`
- Trên **Linux/Mac**: `cp .env.example .env`

Mở file `.env` vừa tạo và chỉnh sửa (nếu cần):
```ini
APP_ENV=development
DEBUG=true
PREDICTION_THRESHOLD=0.5
# Điền API Key của vnstock nếu bạn dùng tính năng pro (nếu không, để trống)
VNSTOCK_API_KEY=
```

---

### 2. Chạy Pipeline Huấn luyện (Training & MLflow)

Thay vì chạy từng file rải rác, hệ thống hiện tại đã được cấu trúc thành Pipeline tự động tự động thu thập (từ vnstock), xử lý, trích xuất đặc trưng và train các mô hình.

#### Cách A: Chạy và trực quan hóa qua Jupyter Notebook (Khuyên dùng cho tính năng tương tác)
1. Khởi động môi trường Notebook:
   ```bash
   jupyter notebook
   ```
2. Trình duyệt sẽ mở ra. Điều hướng tới thư mục `notebooks/` và mở file `00_pipeline_demo.ipynb`.
3. Chạy từng cell (ô lệnh) từ trên xuống dưới (`Shift + Enter`). File này sẽ hướng dẫn bạn cách hàm `run_full_pipeline` hoạt động.

#### Cách B: Chạy qua Python Script
Nếu bạn muốn chạy tự động trên terminal:
1. Tạo một file tên là `run.py` ở thư mục gốc chứa nội dung:
    ```python
    from src.pipeline import run_full_pipeline

    # Tải dữ liệu các mã công ty cụ thể, xử lý và train models.
    # Dữ liệu sẽ được lưu tự động ở ./data và model ở ./models
    artifacts = run_full_pipeline(symbols=["VCB", "FPT", "VIC", "VNM", "SSI"])
    ```
2. Thực thi file:
   ```bash
   python run.py
   ```

#### 📊 Đánh giá & Giám sát với MLflow
Mỗi lần chạy pipeline, hàm `train.py` sẽ tự động ghi lại các metrics (Accuracy, ROC_AUC, F1...) và thông số siêu tham số của toàn bộ mô hình (`XGBoost`, `LightGBM`, `ANN`...) thông qua MLflow.

Để xem kết quả và quyết định mô hình tốt nhất:
1. Mở một terminal mới (nhớ kích hoạt lại `venv`), gõ lệnh:
   ```bash
   mlflow ui --port 5000
   ```
2. Mở trình duyệt và truy cập: [http://localhost:5000](http://localhost:5000)
3. Giao diện quản lý MLflow sẽ hiển thị chi tiết các lần train, bạn có thể click vào để so sánh đồ thị và chọn ra "Best Model". 
*(Hệ thống đã tự động lưu Best Model ra dạng `models/best_model.joblib`)*.

---

### 3. Triển khai API (Model Serving FastAPI)

Khi hàm huấn luyện đã lưu thành công mô hình cùng Scaler và features list vào thư mục `models/`, hệ thống đã sẵn sàng dự đoán thời gian thực.

#### Khởi chạy trực tiếp (Local Test)
Mở terminal và chạy lệnh:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Kiểm tra API có đang hoạt động:** Truy cập [http://localhost:8000/](http://localhost:8000/)
- **Swagger UI (Tài liệu tương tác):** Truy cập [http://localhost:8000/docs](http://localhost:8000/docs). Từ đây bạn có thể ấn "Try it out" để test trực tiếp API bằng giao diện web mà không cần viết code.

#### 🐳 Khởi chạy bằng Docker (Dành cho Production)
Giúp mô phỏng chính xác môi trường khi bạn mang dự án lên cloud (AWS, GCP, Azure).

1. Bạn chỉ cần đảm bảo máy tính đã cài đặt và bật [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Di chuyển vào thư mục deploy:
   ```bash
   cd deploy
   ```
3. Khởi Tải và build Container:
   ```bash
   docker-compose up --build -d
   ```
   *Lệnh này sẽ tự động tải các base image, thiết lập python, cài đặt requirements và khởi chạy.*
   
Hệ thống sẽ chạy song song 2 dịch vụ độc lập trong môi trường ảo hóa:
- **FastAPI Model Serving:** API nhận dự đoán, có sẵn tại `http://localhost:8000`.
- **MLflow Tracking Server:** Máy chủ Tracking độc lập, khả dụng tại `http://localhost:5000`.

*(Muốn dừng toàn bộ server của Docker, dùng lệnh: `docker-compose down`)*

---

## 🛠 Thông tin API (Endpoints) chính

### 1. Lấy trạng thái hệ thống: `GET /health`
Mục đích: Đảm bảo mô hình (Model) và bộ chuẩn hóa dữ liệu (Scaler) đã load thành công trên Server RAM hay chưa.
**CURL Demo:**
```bash
curl -X 'GET' 'http://localhost:8000/health' -H 'accept: application/json'
```

### 2. Dự đoán Gian lận: `POST /predict`
Bạn cần gửi một JSON object chứa chính xác các `feature_cols` mà mô hình đã được train.
**CURL Demo:**
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "Loi_nhuan_sau_thue": 0.15,
  "Tong_tai_san": 0.8,
  "He_so_no": 0.65,
  "ROE": 0.2,
  "Chi_so_A": 1.2,
  "Chi_so_B": 0.95
}'
```
*(Ghi chú: Các trường trong ví dụ bên trên chỉ mang tính minh họa, API thực tế sẽ báo lỗi nêu chính xác danh sách các feature tài chính bị thiếu nếu bạn gửi sai cấu trúc).*

**Đầu ra mong đợi:**
```json
{
  "label": "Fraud",
  "fraud_prediction": 1,
  "fraud_probability": 0.87521,
  "threshold": 0.5,
  "model_name": "XGBClassifier"
}
```

---

## 💻 Tech Stack cốt lõi
- **Data Science / Thuật toán:** `scikit-learn`, `XGBoost`, `LightGBM`, `Pandas`, `NumPy`
- **Data Provider API:** `vnstock` (Lấy dữ liệu cổ phiếu VN)
- **MLOps / Tracking:** `MLflow`
- **Backend / API Framework:** `FastAPI`, `Uvicorn`, `Pydantic` (Data Validation)
- **Deployment & Infra:** `Docker`, `Docker Compose`

---

## 📝 Giấy phép (License)
Dự án được phân phối dưới giấy phép MIT. Vui lòng xem `pyproject.toml` để trích xuất quyền cấp phép chính thức.

