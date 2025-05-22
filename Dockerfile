FROM python:3.9-slim

WORKDIR /app

# Install any needed system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first to ensure it's properly built
RUN pip install --no-cache-dir numpy==1.23.5

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_test_server.py .

CMD ["python", "model_test_server.py"]
