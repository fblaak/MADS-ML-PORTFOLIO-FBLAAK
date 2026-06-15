FROM python:3.12-slim

WORKDIR /app

COPY ingest/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ingest/ingest.py .

CMD ["python", "ingest.py"]