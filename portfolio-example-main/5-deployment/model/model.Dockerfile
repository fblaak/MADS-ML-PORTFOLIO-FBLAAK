FROM python:3.12-slim

WORKDIR /app

COPY model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model/app.py .

EXPOSE 8000

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8000"]