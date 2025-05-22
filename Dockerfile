FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY permission_check_server.py .

CMD ["python", "table_check_server.py"]
