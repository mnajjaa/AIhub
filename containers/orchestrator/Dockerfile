FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY orchestrator/orchestrator.py /app/orchestrator.py

EXPOSE 8000
CMD ["uvicorn", "orchestrator:app", "--host", "0.0.0.0", "--port", "8000"]
