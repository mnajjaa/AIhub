# Agentic App

A lightweight agent runner with MLflow tracking, a CLI, and an optional FastAPI service.

## Setup

```bash
python -m venv .venv
```

Activate the virtual environment:

```bash
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1


Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables (recommended via `.env`):

```bash
OPENAI_API_KEY=your-key-here
# Optional: defaults to sqlite:///mlflow.db
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

## Run an Agent (CLI)

Generate an agent scaffold:

```bash
python agent_cli.py generate --agent_type summarizer
```

Run the agent with inline JSON input:

```bash
python agent_cli.py run --agent_type summarizer --input '{"text": "Hello world"}'
```

Run with JSON from a file:

```bash
python agent_cli.py run --agent_type summarizer --input input.json
```

## Run the API (Optional)

```bash
uvicorn app:app --reload
```

Then POST JSON to the `/run/{agent_name}` endpoint.

## Testing

There is no automated test suite yet. Use a CLI smoke test as a quick check:

```bash
python agent_cli.py run --agent_type summarizer --input '{"text": "Smoke test"}'
```

If you add tests later (for example with pytest), run:

```bash
python -m pytest
```

## View MLflow

Start the MLflow UI pointing at the local SQLite DB:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open the UI in your browser:

```text
http://localhost:5000
```

Run IDs and input hashes are logged by the agent on start and completion.
