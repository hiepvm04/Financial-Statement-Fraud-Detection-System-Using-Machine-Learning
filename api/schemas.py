from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    DSRI: float = Field(..., description="Days Sales in Receivables Index")
    GMI: float = Field(..., description="Gross Margin Index")
    AQI: float = Field(..., description="Asset Quality Index")
    SGI: float = Field(..., description="Sales Growth Index")
    DEPI: float = Field(..., description="Depreciation Index")
    SGAI: float = Field(..., description="SG&A Index")
    LVGI: float = Field(..., description="Leverage Index")
    TATA: float = Field(..., description="Total Accruals to Total Assets")
    RSST_Accruals: float = Field(..., description="RSST Accruals")
    Delta_Receivables: float = Field(..., description="Change in Receivables")
    Delta_Inventory: float = Field(..., description="Change in Inventory")
    Delta_Cash_Sales: float = Field(..., description="Change in Cash Sales")


class PredictionOutput(BaseModel):
    label: Literal["Fraud", "Non-Fraud"]
    fraud_prediction: int = Field(..., ge=0, le=1)
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_name: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    scaler_loaded: bool
    feature_count: int
    model_name: str