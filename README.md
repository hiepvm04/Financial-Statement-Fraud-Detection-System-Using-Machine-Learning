<div align="center">

# 📊 Financial Statement Fraud Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-00a393.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.2+-blue.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-blue?logo=lightgbm)](https://lightgbm.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

*An end-to-end MLOps system for detecting financial statement fraud in the Vietnamese stock market.*

</div>

---

# Overview

The **Financial Statement Fraud Detection System** is an end-to-end machine learning solution designed to analyze financial statements and detect potential fraud in companies listed on the Vietnamese stock market.

The system follows **modern MLOps practices**, automating the full pipeline from:

- Data extraction (ETL via `vnstock`)
- Data preprocessing and cleaning
- Feature engineering
- Training multiple machine learning models
- Experiment tracking with **MLflow**
- Model deployment via **FastAPI**
- Containerized deployment with **Docker**

This project demonstrates a **production-ready machine learning workflow** for financial fraud detection.

---

# Key Features

### Automated ETL Pipeline
- Collects financial statement data directly from `vnstock`
- Supports:
  - Balance Sheet
  - Income Statement
  - Cash Flow
- Automatically computes financial fraud indicators such as:
  - DSRI
  - AQI
  - Accrual-based ratios
  - Soft asset ratios

---

### Advanced Feature Engineering
The pipeline includes several advanced feature selection techniques:

- Variance filtering (`VarianceThreshold`)
- Multicollinearity removal
- Feature selection using **Mutual Information**

These techniques help improve model performance and reduce noise.

---

### Multi-Model Training
The training pipeline automatically trains and compares multiple machine learning models:

- Logistic Regression
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Artificial Neural Network (MLPClassifier)

The system performs **hyperparameter tuning** and selects the best-performing model.

---

### MLflow Experiment Tracking
The project integrates **MLflow** for full experiment tracking:

- Logs training parameters
- Tracks evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
- Stores model artifacts
- Enables experiment comparison

---

### FastAPI Model Serving
The trained model is deployed through a **FastAPI REST API**, enabling:

- Real-time fraud prediction
- Low-latency model inference
- Interactive API documentation (Swagger UI)

---

### Dockerized Deployment
The entire system can be deployed using **Docker + Docker Compose**, which includes:

- FastAPI server
- MLflow tracking server

This allows easy local or production deployment.

---

# 📁 Project Structure

```text
Fraud-Detection/
│
├── api/                  # FastAPI Application (Model Serving)
│   ├── main.py           # FastAPI configuration and endpoints
│   ├── schemas.py        # Pydantic models for input/output validation
│   └── services.py       # Model loading and prediction logic
│
├── data/                 # Raw and processed datasets
│
├── deploy/               # Deployment configuration
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── models/               # Stored artifacts (trained models, scalers, features)
│
├── notebooks/            # Jupyter notebooks for exploration and EDA
│
├── src/                  # Core ML pipeline source code
│   ├── config.py         # Global configuration
│   ├── data.py           # Data collection and ETL logic
│   ├── preprocessing.py  # Data cleaning
│   ├── features.py       # Feature engineering
│   ├── train.py          # Model training
│   ├── evaluate.py       # Model evaluation and metrics
│   └── pipeline.py       # End-to-end pipeline execution
│
├── tests/                # Unit and system tests
│
├── pyproject.toml        # Project dependencies
├── .env.example          # Environment configuration
└── README.md
```

---

## ⚙️ Installation & Getting Started

### System Requirements (Prerequisites)
- **Operating System:** Windows, macOS, or Linux
- **Python:** `>= 3.10`
- **Docker & Docker Compose** (optional, for production deployment)

---

### 1️⃣ Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/hiepvm04/Financial-Statement-Fraud-Detection-Using-Machine-Learning.git
cd Fraud-Detection

# 2. Create and activate a virtual environment
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate

# 3. Install all dependencies (API, notebooks, testing libraries)
pip install -e .[all]

# 4. Configure environment variables
cp .env.example .env
# Update VNSTOCK_API_KEY in the .env file if required
```
### 2️⃣ Run the Training Pipeline

The project provides an integrated pipeline that executes the full workflow:

**Data Collection → Data Cleaning → Feature Engineering → Model Training → Model Saving**

Run the pipeline with:

```bash
python -m scripts.run_pipeline
```
---

## API Deployment (Model Serving)

After the training pipeline is completed and the **best model has been saved**, the API is ready to serve predictions.

### Run Locally

Start the FastAPI server:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Live API Endpoint:** [http://localhost:8000](http://localhost:8000)
- **Interactive API Docs (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)

### Production Deployment with Docker

The Docker environment runs both the FastAPI server and the MLflow tracking server simultaneously.
```bash
cd deploy
docker-compose up --build -d
```
The API will listen on port `:8000` and MLflow will run on port `:5000`.  
To stop all running services, use:

```bash
docker-compose down
```
---
## API Reference

### 1️⃣ System Health Check: `GET /health`

```bash
curl -X 'GET' 'http://localhost:8000/health' -H 'accept: application/json'
```
*Expected output:*
```json
{
  "status": "ok",
  "model_loaded": true,
  "scaler_loaded": true,
  "feature_count": 12,
  "model_name": "XGBClassifier"
}
```

### 2️⃣ Fraud Prediction Request: `POST /predict`
The request body must contain the exact `features` selected during the training stage.
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
*Note: The number of input features may vary depending on the automatic feature selection configuration used in the training pipeline.*

*Expected output:*
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

## Tech Stack
- **Machine Learning Data Processing:** `Pandas`, `NumPy`, `Scikit-Learn`
- **Core ML Algorithms:** `XGBoost`, `LightGBM`, `Scikit-Learn (SVM, Logsitic Regression, ANN)`
- **Data Source API:** `vnstock` 
- **Model Monitoring:** `MLflow` 
- **Backend & Web API:** `FastAPI`, `Uvicorn`, `Pydantic`
- **Infrastructure:** `Docker`, `Docker-Compose`

---

## Contact Information

This project is a personal product for coursework and practical research. If you find it useful or have suggestions for improving the code/architecture, please open an Issue or reach out via:

- **Author:** Vu Manh Hiep
- **Email:** hiepvm04@gmail.com
- **LinkedIn:** 
- **Github:** https://github.com/hiepvm04
