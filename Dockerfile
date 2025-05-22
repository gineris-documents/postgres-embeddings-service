FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY debug_server.py .
COPY db_test_server.py .
COPY simple_server.py .

# List all files for debugging
RUN ls -la

CMD ["python", "debug_server.py"]
