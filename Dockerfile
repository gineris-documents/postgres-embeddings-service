FROM python:3.9-slim

WORKDIR /app

# Install system packages including Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libc6-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Install numpy first to ensure it's properly built
RUN pip install --no-cache-dir numpy==1.23.5

# Install Google libraries explicitly
RUN pip install --no-cache-dir google-api-python-client==2.41.0 google-auth==2.6.0 google-auth-httplib2==0.1.0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY document_processor.py .

# Create an empty service-account.json file if it doesn't exist
RUN touch service-account.json

# Verify installations
RUN pip list | grep -E "(google|torch|transformers|pymupdf|pytesseract)"

# Test Tesseract installation
RUN tesseract --version

CMD ["python", "document_processor.py"]
