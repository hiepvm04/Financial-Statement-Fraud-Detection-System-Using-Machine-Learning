# ===============================
# Base Image
# ===============================
FROM python:3.11-slim

# ===============================
# Environment variables
# ===============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ===============================
# System dependencies
# ===============================
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# Work directory
# ===============================
WORKDIR /app

# ===============================
# Copy dependency files
# ===============================
COPY pyproject.toml /app/

# Install dependencies
RUN pip install --upgrade pip \
    && pip install .

# ===============================
# Copy project files
# ===============================
COPY . /app

# ===============================
# Expose API port
# ===============================
EXPOSE 8000

# ===============================
# Start API
# ===============================
CMD ["bash", "deploy/serve_model.sh"]