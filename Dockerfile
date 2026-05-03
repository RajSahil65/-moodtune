FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends 
    libgl1 
    libglib2.0-0 
    libsm6 
    libxext6 
    libxrender1 
    libgomp1 
    libsndfile1 
    ffmpeg 
    curl 
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && 
    pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY frontend/ ./frontend/

WORKDIR /app/backend

CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
