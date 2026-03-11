# Financial Statement Fraud Detection

Dự án này sử dụng Học máy (Machine Learning) để phát hiện gian lận trong báo cáo tài chính của các công ty niêm yết trên thị trường chứng khoán Việt Nam.

Dự án đã được tái cấu trúc theo hướng **MLOps** để dễ dàng mở rộng, huấn luyện, theo dõi thí nghiệm (MLflow), và đưa vào thực tế (Deployment).

## 🚀 Cấu trúc thư mục hiện tại (Giai đoạn 1 & 2)

```text
Fraud-Detection/
│
├── data/                           # Dữ liệu cục bộ (raw và processed)
├── notebooks/                      # Chứa Jupyter Notebooks để demo và phân tích
│   ├── 00_pipeline_demo.ipynb      # Điểm bắt đầu: Demo cách dùng các hàm từ src/
│   └── ...                         # Các notebooks cũ
│
├── src/                            # Source code chính của dự án 
│   ├── config.py                   # Cấu hình đường dẫn, hằng số toàn cục
│   ├── data.py                     # Hàm tải dữ liệu từ vnstock
│   ├── preprocessing.py            # Hàm xử lý missing values, làm sạch dữ liệu
│   ├── features.py                 # Hàm tạo đặc trưng tài chính (Feature Engineering)
│   ├── train.py                    # Huấn luyện mô hình và Tracking với MLflow
│   └── evaluate.py                 # ĐÁnh giá mô hình, vẽ Confusion Matrix, ROC
│
├── .env                            # Tệp chứa biến môi trường (port, uri...)
└── requirements.txt                # Danh sách thư viện cần cài đặt
```

## 🛠 Hướng dẫn Cài đặt & Chạy Code

### 1. Cài đặt thư viện
Bạn nên tạo một môi trường ảo (virtual environment) trước khi cài đặt để tránh xung đột thư viện.
Mở Terminal / PowerShell tại thư mục `Fraud-Detection` và chạy:

```bash
# Cài đặt toàn bộ thư viện cần thiết
pip install -r requirements.txt
```

### 2. Chạy thử nghiệm với Notebook Demo
Cách tốt nhất để hiểu luồng đi của dữ liệu hiện tại là sử dụng tệp Demo Notebook đã được chuẩn bị sẵn.

```bash
# Khởi động Jupyter Notebook
jupyter notebook
```
- Mở tệp `notebooks/00_pipeline_demo.ipynb`.
- Chạy từng cell từ trên xuống dưới. Notebook này sẽ biểu diễn cách:
  1. Tải dữ liệu bằng `vnstock` thông qua `src.data`.
  2. Làm sạch và tạo đặc trưng bằng `src.preprocessing` và `src.features`.
  3. Huấn luyện mô hình XGBoost và lưu lại lịch sử với `src.train`.

### 3. Theo dõi mô hình bằng MLflow
Trong quá trình chạy thuật toán huấn luyện ở `src.train.py`, hệ thống sẽ tự động ghi lại lịch sử huấn luyện (độ chính xác, tham số, file mô hình) vào thư mục cục bộ `mlruns/`.

Để xem giao diện quản lý thí nghiệm của MLflow, hãy mở một Terminal mới tại thư mục gốc của dự án và chạy:

```bash
mlflow ui --port 5000
```
Sau đó, mở trình duyệt và truy cập vào địa chỉ: `http://localhost:5000`

---

*Lưu ý: Hệ thống hiện tại đang hoàn thiện Giai đoạn 1 (Setup Cấu trúc) và Giai đoạn 2 (Core ML Modules). Giai đoạn tiếp theo sẽ bao gồm xây dựng API (FastAPI) và đóng gói Docker.*
