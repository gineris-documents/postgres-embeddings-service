FROM python:3.9-slim

WORKDIR /app

COPY minimal_server.py .

CMD ["python", "minimal_server.py"]
