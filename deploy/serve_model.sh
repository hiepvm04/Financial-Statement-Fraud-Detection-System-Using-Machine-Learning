#!/bin/bash

set -e

echo "Starting Fraud Detection API..."

HOST=${API_HOST:-0.0.0.0}
PORT=${API_PORT:-8000}
WORKERS=${API_WORKERS:-1}

echo "Host: $HOST"
echo "Port: $PORT"
echo "Workers: $WORKERS"

uvicorn api.main:app \
  --host $HOST \
  --port $PORT \
  --workers $WORKERS