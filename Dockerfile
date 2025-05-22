FROM python:3.9-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create an empty service-account.json file if it doesn't exist
RUN touch service-account.json

# Make sure the app listens on the port provided by Cloud Run
CMD ["python", "app.py"]
