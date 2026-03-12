from __future__ import annotations

import uvicorn

from src.config import API_HOST, API_PORT


def main() -> None:
    print("Starting FastAPI server...")
    print(f"Host: {API_HOST}")
    print(f"Port: {API_PORT}")

    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )


if __name__ == "__main__":
    main()