# MLflow Integration - Quick Start Guide

## Installation

```bash
pip install mlflow
```

## Key Files Added/Modified

| File | Type | Purpose |
|------|------|---------|
| `artifact_manager.py` | NEW | Load artifacts from MLflow with caching |
| `mlflow_utils.py` | NEW | MLflow experiment tracking utilities |
| `config_loader.py` | UPDATED | Unified config loading with artifact support |
| `agent_core.py` | UPDATED | Instrumented with MLflow logging |
| `app.py` | UPDATED | REST API MLflow integration with request IDs |
| `agent_cli.py` | UPDATED | CLI MLflow integration |
| `requirements.txt` | UPDATED | Added mlflow==2.11.0 |

## Quick Examples

### Log Agent Artifacts
```bash
python agent_cli.py log-artifacts --agent_type summarizer --version v1 --update_config
```

### Load Artifact from MLflow
```python
from artifact_manager import ArtifactManager

manager = ArtifactManager()
prompt = manager.load_artifact("mlflow://prompts/summarizer/system_v1")
schema = manager.load_json_artifact("mlflow://schemas/summarizer/output_v1")
```

### Run Agent with Tracking (CLI)
```bash
python agent_cli.py run --agent_type summarizer --input '{"text": "..."}'
```

### Run Agent with Tracking (API)
```python
import uvicorn
# Start server
uvicorn.run("app:app", reload=True)

# Then POST to http://localhost:8000/run/summarizer
```

### View Experiments
```bash
mlflow ui --host 0.0.0.0 --port 5000
# Visit http://localhost:5000
```

### Query Runs Programmatically
```python
from mlflow import MlflowClient

client = MlflowClient()
experiment = client.get_experiment_by_name("AgentRuns")
runs = client.search_runs(experiment_ids=[experiment.experiment_id])

for run in runs:
    print(f"Run {run.info.run_id}: {run.data.tags.get('status')}")
    print(f"  Metrics: {run.data.metrics}")
    print(f"  Params: {run.data.params}")
```

## MLflow URI Format

```
mlflow://collection/item/version

Examples:
  mlflow://prompts/summarizer/system_v1
  mlflow://schemas/researcher/output_v2
  mlflow://configs/researcher/default_v1
```

## Environment Variables

```bash
# Set tracking server location (default: sqlite:///mlflow.db)
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

# Set registry server (optional)
export MLFLOW_REGISTRY_URI="sqlite:///mlflow.db"
```

## Tracked Metrics & Parameters

### Every Run Logs:
- **Tags**: entry_point, agent_name, status, parse_stage
- **Params**: agent_name, model_name, temperature, input_hash, parse_final_stage, error_message
- **Metrics**: input_field_count, tokens_*, api_latency_ms, parse_attempts_needed, validation_errors_count, total_latency_ms

### Example MLflow UI View:
```
Experiment: AgentRuns

Run 1 (summarizer_20260126_143022)
├── Tags
│   ├── entry_point: rest_api
│   ├── agent_name: summarizer
│   ├── status: success
│   └── parse_stage: original
├── Parameters
│   ├── agent_name: summarizer
│   ├── model_name: gpt-3.5-turbo
│   └── input_hash: 4a8f9e1d2c5b3a7f...
└── Metrics
    ├── total_latency_ms: 2345.67
    ├── tokens_prompt: 154
    ├── tokens_completion: 89
    ├── tokens_total: 243
    ├── parse_attempts_needed: 1
    └── output_field_count: 1
```

## Configuration Examples

### Agent Config with MLflow URIs
```yaml
agent_type: "summarizer"
model:
  provider: "openai"
  model_name: "gpt-3.5-turbo"
  temperature: 0
prompts:
  system: "mlflow://prompts/summarizer/system_v1"
  task: "mlflow://prompts/summarizer/task_v1"
input_schema: "mlflow://schemas/summarizer/input_v1"
output_schema: "mlflow://schemas/summarizer/output_v1"
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  run_id: "REPLACE_WITH_RUN_ID"  # Optional for run-based artifact URIs
  fallback_paths:
    "mlflow://prompts/summarizer/system_v1": "prompts/summarizer_system.txt"
    "mlflow://prompts/summarizer/task_v1": "prompts/summarizer_task.txt"
    "mlflow://schemas/summarizer/input_v1": "schemas/summarizer_input.json"
    "mlflow://schemas/summarizer/output_v1": "schemas/summarizer_output.json"
```

### Backward Compatible (Local Paths Still Work)
```yaml
agent_type: "summarizer"
model:
  provider: "openai"
  model_name: "gpt-3.5-turbo"
  temperature: 0
prompts:
  system: "prompts/summarizer_system.txt"
  task: "prompts/summarizer_task.txt"
input_schema: "schemas/summarizer_input.json"
output_schema: "schemas/summarizer_output.json"
```

## Graceful Degradation

If MLflow is not installed:
- ✅ CLI and API continue to work
- ✅ Local file loading works
- ✅ Agent execution proceeds without tracking
- ✅ No runtime errors (just warnings)

```python
# Works even without MLflow
python agent_cli.py run --agent_type summarizer --input '{"text": "..."}'
# Output: ✅ Agent execution successful!
```

## Integration Points

### 1. CLI (`agent_cli.py`)
- Prints MLflow run ID after execution
- Shows MLflow UI URL for viewing results
- Logs prompts/schemas/config with `log-artifacts`

### 2. REST API (`app.py`)
- Generates request ID for traceability
- Includes `_mlflow_run_id` in response
- Logs API entry point context

### 3. Agent Execution (`agent_core.py`)
- Logs 10 instrumentation points:
  1. Run entry (agent name, input hash)
  2. Config loading (model, temperature)
  3. OpenAI API call params
  4. OpenAI response (tokens, latency)
  5. JSON parsing stage
  6. Output validation
  7. Output artifact logging
  8. Successful completion
  9. Error handling
  10. Run finalization

### 4. Artifact Management (`artifact_manager.py`)
- Handles mlflow:// URIs transparently
- Falls back to local files
- Implements caching

### 5. Config Loading (`config_loader.py`)
- Detects mlflow:// scheme
- Routes to artifact manager or filesystem

## Troubleshooting

### "MLflow not available" message
```bash
pip install mlflow
```

### Cannot find artifact in MLflow
- Check artifact registered in MLflow Registry
- Verify URI format is correct: `mlflow://collection/item/version`
- Check MLflow tracking server is running

### Run not appearing in MLflow UI
```bash
# Start MLflow UI if not running
mlflow ui --host 0.0.0.0 --port 5000

# Check tracking URI
echo $MLFLOW_TRACKING_URI
# Should show: sqlite:///mlflow.db (or your configured URI)
```

### Clear MLflow Cache
```python
from artifact_manager import ArtifactManager
manager = ArtifactManager()
manager.clear_cache(max_age_days=1)  # Remove 1+ day old cache
```

## Next: Advanced Usage

1. **Register Models in MLflow Model Registry**
   ```python
   client = MlflowClient()
   client.create_registered_model("my_model")
   ```

2. **Set Up Remote MLflow Server**
   ```bash
   mlflow server --backend-store-uri postgresql://user:pwd@localhost/mlflow \
                 --default-artifact-root s3://my-bucket/mlflow
   ```

3. **Query and Analyze Runs**
   ```python
   client.search_runs(filter_string="metrics.total_latency_ms < 5000")
   ```

4. **Export Runs for Analysis**
   ```python
   import pandas as pd
   runs_df = pd.DataFrame([run.to_dictionary() for run in runs])
   ```

---

**For detailed documentation**, see: [MLFLOW_INTEGRATION_SUMMARY.md](MLFLOW_INTEGRATION_SUMMARY.md)
