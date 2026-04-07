FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# El script principal entrena todos los modelos y guarda en /app/models
ENV MODEL_DIR=/app/models
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

CMD ["python", "run_all_training.py"]