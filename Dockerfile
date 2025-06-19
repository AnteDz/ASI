FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml poetry.lock* ./
COPY src/ ./src
COPY app.py ./

COPY data/01_raw           ./data/01_raw
COPY data/02_intermediate  ./data/02_intermediate
COPY data/06_models        ./data/06_models
COPY data/07_model_output  ./data/07_model_output

EXPOSE 8501
ENTRYPOINT ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
