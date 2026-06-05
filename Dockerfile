FROM python:3.10-slim

# 시스템 패키지 설치 (OpenCV 헤드리스용)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# requirements 먼저 복사 (캐시 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 파일 복사
COPY . .

# Hugging Face Spaces는 7860 포트 사용
EXPOSE 7860

# Flask 앱 실행 (gunicorn으로 안정적 실행)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:app"]
