from __future__ import annotations

import requests

from src.config import API_HOST, API_PORT


def main() -> None:
    base_url = f"http://{API_HOST}:{API_PORT}".replace("0.0.0.0", "127.0.0.1")
    health_url = f"{base_url}/health"
    predict_url = f"{base_url}/predict"

    payload = {
        "DSRI": 1.12,
        "GMI": 0.98,
        "AQI": 1.05,
        "SGI": 1.20,
        "DEPI": 0.95,
        "SGAI": 1.08,
        "LVGI": 1.10,
        "TATA": 0.03,
        "RSST_Accruals": 0.02,
        "Delta_Receivables": 0.01,
        "Delta_Inventory": 0.02,
        "Delta_Cash_Sales": 0.15,
    }

    print("=== HEALTH CHECK ===")
    health_response = requests.get(health_url, timeout=30)
    print("Status code:", health_response.status_code)
    print("Response:", health_response.json())

    print("\n=== PREDICT ===")
    predict_response = requests.post(predict_url, json=payload, timeout=30)
    print("Status code:", predict_response.status_code)
    print("Response:", predict_response.json())


if __name__ == "__main__":
    main()