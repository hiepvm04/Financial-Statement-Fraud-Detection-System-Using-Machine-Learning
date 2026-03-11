from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PredictionInput, PredictionOutput
from api.services import model_service
from src.config import APP_NAME, APP_ENV, DEBUG


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model_service.load_artifacts()
        print("Model artifacts loaded successfully.")
    except Exception as e:
        print(f"Failed to load model artifacts: {e}")
    yield


app = FastAPI(
    title=APP_NAME,
    version="0.1.0",
    description="API for Financial Statement Fraud Detection",
    debug=DEBUG,
    lifespan=lifespan,
)


@app.get("/", tags=["Root"])
def root() -> dict:
    return {
        "message": "Financial Statement Fraud Detection API",
        "environment": APP_ENV,
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    return HealthResponse(
        status="ok" if model_service.is_loaded else "degraded",
        model_loaded=model_service.model is not None,
        scaler_loaded=model_service.scaler is not None,
        feature_count=len(model_service.feature_cols),
        model_name=model_service.model_name,
    )


@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict(payload: PredictionInput):
    if not model_service.is_loaded:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded.")

    try:
        result = model_service.predict(payload.model_dump())
        return PredictionOutput(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}") from e