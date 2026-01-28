FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app.py agent_core.py config_loader.py artifact_manager.py mlflow_utils.py /app/
COPY configs/researcher.yaml /app/configs/
COPY prompts/researcher_* /app/prompts/
COPY schemas/researcher_* /app/schemas/

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
