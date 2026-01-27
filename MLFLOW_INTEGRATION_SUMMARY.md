# MLflow Integration Summary

## Overview

Successfully integrated MLflow into the agentic-app project to provide:
- Centralized artifact registry for prompts, schemas, and configurations
- Experiment tracking for each agent invocation
- Versioning support via `mlflow://` URI scheme
- Full backward compatibility with local file paths

## Components Implemented

### 1. **artifact_manager.py** - MLflow Artifact Loading
- `ArtifactManager` class handles loading from both MLflow URIs and local files
- Supports URI format: `mlflow://collection/item/version`
- Supports run ID mapping, explicit artifact URI mapping, and model registry fallback
- Supports optional local fallback paths when MLflow is unavailable or an artifact is missing
- Implements artifact caching with configurable TTL

**Key Features:**
```python
manager = ArtifactManager(tracking_uri="sqlite:///mlflow.db")
content = manager.load_artifact("mlflow://prompts/summarizer/system_v1")
json_data = manager.load_json_artifact("mlflow://schemas/summarizer/output_v1")
```

### 2. **config_loader.py** - Unified Configuration Loading
- `ConfigLoader` class wraps artifact resolution
- `resolve_artifact_path()` - Load from MLflow or local filesystem
- Applies optional `mlflow` settings from agent config (tracking_uri, run_id, mappings, fallbacks)
- Convenience functions: `load_config()`, `load_prompt()`, `load_schema()`
- Global singleton instance for consistent access

**Key Features:**
```python
loader = ConfigLoader()
prompt = loader.load_prompt("mlflow://prompts/summarizer/system_v1")
schema = loader.load_schema("mlflow://schemas/summarizer/input_v1")
```

### 3. **mlflow_utils.py** - Experiment Tracking Utilities
Provides comprehensive MLflow tracking functions with graceful fallback when MLflow is unavailable:

- `setup_mlflow_tracking()` - Initialize tracking server
- `log_openai_call_params()` - Log model parameters and prompts
- `log_openai_response()` - Log token usage and API latency
- `log_json_parsing()` - Log JSON parsing stages and recovery info
- `log_output_validation()` - Log validation metrics
- `log_agent_output()` - Log output JSON as an artifact
- `log_agent_artifacts()` - Log prompts/schemas/config as versioned artifacts
- `log_execution_success()` - Log successful completion
- `log_execution_error()` - Log errors with traceback
- `end_agent_run()` - Finalize MLflow run
- `get_run_url()` - Generate MLflow UI URL
- `get_run_info()` - Retrieve run metadata

**All functions check `MLFLOW_AVAILABLE` and gracefully handle when MLflow is not installed.**

### 4. **agent_core.py** - MLflow-Instrumented Agent Execution
Updated to track all execution stages:

**Instrumentation Points:**
1. **Run Entry** - Log agent name, input hash, entry point
2. **Config Loading** - Log model name, temperature
3. **OpenAI Call** - Log API parameters and token usage
4. **JSON Parsing** - Log parse stage (original/retry/repair), attempts needed
5. **Output Validation** - Log field counts, validation errors
6. **Output Artifact** - Log output JSON payload as an artifact
7. **Run Completion** - Log success status and total latency
8. **Error Handling** - Log error type, message, and traceback

**Lazy MLflow Loading:**
- MLflow utilities imported only when needed (inside `run_agent()`)
- Supports execution without MLflow installed
- No breaking changes to existing functionality

### 5. **app.py** - REST API MLflow Integration
- Request ID generation for traceability
- MLflow run context for each `/run/{agent_name}` POST request
- Run ID included in response as `_mlflow_run_id`
- Graceful fallback when MLflow unavailable (`_request_id` instead)

### 6. **agent_cli.py** - CLI MLflow Integration
- Optional MLflow run tracking for CLI commands
- Print MLflow run ID and UI link after execution
- `log-artifacts` subcommand to log prompts/schemas/config and optionally update configs
- MLflow import is conditional (try/except)
- Gracefully handles missing MLflow

## Configuration

### Environment Variables
```bash
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"  # Default
export MLFLOW_REGISTRY_URI="sqlite:///mlflow.db"  # Optional
```

### MLflow URI Format
```
mlflow://collection/item/version

Examples:
- mlflow://prompts/summarizer/system_v1
- mlflow://prompts/researcher/task_v2
- mlflow://schemas/summarizer/input_v1
- mlflow://schemas/researcher/output_v3
```

## Updated Configuration Files

### Example Agent Config with MLflow URIs
```yaml
agent_type: "summarizer"

model:
  provider: "openai"
  model_name: "gpt-3.5-turbo"
  temperature: 0

prompts:
  system: "mlflow://prompts/summarizer/system_v1"    # MLflow artifact
  task: "mlflow://prompts/summarizer/task_v1"        # MLflow artifact

input_schema: "mlflow://schemas/summarizer/input_v1"   # MLflow artifact
output_schema: "mlflow://schemas/summarizer/output_v1" # MLflow artifact

tools: []

mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  run_id: "REPLACE_WITH_RUN_ID"  # Optional for run-based artifact URIs
  fallback_paths:
    "mlflow://prompts/summarizer/system_v1": "prompts/summarizer_system.txt"
    "mlflow://prompts/summarizer/task_v1": "prompts/summarizer_task.txt"
    "mlflow://schemas/summarizer/input_v1": "schemas/summarizer_input.json"
    "mlflow://schemas/summarizer/output_v1": "schemas/summarizer_output.json"
```

### Backward Compatibility
Existing configs with local paths continue to work:
```yaml
prompts:
  system: "prompts/summarizer_system.txt"    # Still supported
  task: "prompts/summarizer_task.txt"

input_schema: "schemas/summarizer_input.json"
output_schema: "schemas/summarizer_output.json"
```

Configs that use `mlflow://` URIs can also provide `mlflow.fallback_paths` to keep local files as a fallback.

## Tracked Metrics

### Execution Metrics
- `total_latency_ms` - Total execution time
- `input_field_count` - Number of input fields
- `output_field_count` - Number of output fields  
- `output_size_bytes` - Size of output JSON

### OpenAI Metrics
- `tokens_prompt` - Prompt tokens used
- `tokens_completion` - Completion tokens used
- `tokens_total` - Total tokens
- `openai_api_latency_ms` - API call latency
- `response_length_chars` - Response character count

### Parsing Metrics
- `parse_attempts_needed` - Number of attempts (1, 2, or 3)
- `parse_recovery_required` - Whether recovery was needed (1 if retry/repair)
- `parsed_json_field_count` - Fields in parsed JSON
- `parsed_json_size_bytes` - Size of parsed JSON

### Validation Metrics
- `validation_errors_count` - Number of validation errors

## Logged Parameters

- `agent_name` - Agent identifier
- `model_name` - LLM model used
- `temperature` - Temperature parameter
- `input_hash` - SHA-256 hash of input
- `input_hash` - SHA-256 hash of input data
- `parse_final_stage` - Final parsing stage reached
- `output_fields` - List of output field names
- `error_message` - Truncated error message (if failed)

## Logged Tags

- `entry_point` - How agent was invoked (cli, rest_api, agent_core)
- `agent_name` - Agent type/name
- `status` - Execution result (success, error)
- `parse_stage` - Final JSON parsing stage
- `error_type` - Exception class name (if failed)
- `error_stage` - Stage where error occurred
- `execution_timestamp` - ISO timestamp of run start

## Installation

### Add MLflow to Your Environment
```bash
pip install mlflow
```

### Requirements Updated
Added `mlflow==2.11.0` to [requirements.txt](requirements.txt)

## Usage Examples

### Log Agent Artifacts to MLflow
```bash
# Log prompts/schemas/config and update config with mlflow:// URIs
python agent_cli.py log-artifacts --agent_type summarizer --version v1 --update_config
```

### CLI with MLflow Tracking
```bash
# MLflow will track this execution
python agent_cli.py run --agent_type summarizer --input '{"text": "..."}'

# Check MLflow UI
mlflow ui --host 0.0.0.0 --port 5000
```

### REST API with MLflow Tracking
```bash
# Start the server
python -m uvicorn app:app --reload

# Make request
curl -X POST http://localhost:8000/run/summarizer \
  -H "Content-Type: application/json" \
  -d '{"text": "Machine learning..."}'

# Response includes MLflow run ID:
{"summary": "...", "_mlflow_run_id": "a1b2c3d4-e5f6-..."}
```

### Load Artifacts from MLflow
```python
from artifact_manager import ArtifactManager

manager = ArtifactManager(tracking_uri="sqlite:///mlflow.db")

# Load text artifact
system_prompt = manager.load_artifact("mlflow://prompts/summarizer/system_v1")

# Load JSON artifact
schema = manager.load_json_artifact("mlflow://schemas/summarizer/output_v1")

# Clear old cache
manager.clear_cache(max_age_days=30)
```

## Backward Compatibility

✅ **Fully backward compatible:**
- Existing configs with local file paths work unchanged
- MLflow is optional (graceful degradation if not installed)
- All new features are non-breaking
- Lazy loading prevents import errors when MLflow unavailable

## Graceful Degradation

When MLflow is not installed:
- ✅ CLI and API still function normally
- ✅ Artifact loading from local files works
- ✅ Agent execution proceeds without tracking
- ✅ Useful warning message displayed
- ✅ No runtime errors

## Next Steps

1. **Log Artifacts and Update Configs:**
   ```bash
   python agent_cli.py log-artifacts --agent_type summarizer --version v1 --update_config
   ```

2. **Set Up MLflow Server (Optional):**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db \
                 --default-artifact-root ./artifacts \
                 --host 0.0.0.0 --port 5000
   ```

3. **Register Assets in Model Registry (Optional):**
   - Create registered models for prompt/schema assets
   - Use model aliases or stages for `mlflow://` resolution

4. **Query Experiment Results:**
   ```python
   from mlflow import MlflowClient
   client = MlflowClient()
   runs = client.search_runs(experiment_names=["AgentRuns"])
   ```

## Testing

Run without MLflow installed:
```bash
pip uninstall mlflow -y
python agent_cli.py --help  # Works fine
```

Install and run with MLflow:
```bash
pip install mlflow
python agent_cli.py run --agent_type summarizer --input '{"text": "..."}'
# View runs: mlflow ui
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                      │
├─────────────────────────────────────────────────────────┤
│  CLI (agent_cli.py) │ REST API (app.py)                 │
└───────────┬─────────────────┬──────────────────────────┘
            │                 │
            ├─────────────────┴──────────────┐
                              │
                    ┌─────────▼──────────┐
                    │  agent_core.py     │
                    │  (Main Execution)  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────────────┐
                    │  config_loader.py          │
                    │  (Config + Artifact Mgmt)  │
                    └─────────┬──────────────────┘
                              │
                ┌─────────────┴──────────────┐
                │                            │
        ┌───────▼─────────┐      ┌──────────▼─────────┐
        │artifact_manager │      │ Local Filesystem   │
        │ (MLflow Client) │      │  (prompts/,        │
        │                 │      │   configs/,        │
        └────────┬────────┘      │   schemas/)        │
                 │               └────────────────────┘
        ┌────────▼──────────────────┐
        │   MLflow Tracking Server  │
        │  (sqlite:///mlflow.db)    │
        │  - Experiments            │
        │  - Runs                   │
        │  - Metrics                │
        │  - Artifacts              │
        └───────────────────────────┘
```

## File Structure

```
agentic-app/
├── artifact_manager.py          # NEW: MLflow artifact loading
├── mlflow_utils.py              # NEW: MLflow tracking utilities
├── config_loader.py             # UPDATED: Unified config + artifact loading
├── agent_core.py                # UPDATED: MLflow instrumentation
├── agent_cli.py                 # UPDATED: CLI MLflow integration
├── app.py                        # UPDATED: REST API MLflow integration
├── requirements.txt             # UPDATED: Added mlflow==2.11.0
├── configs/
│   ├── researcher.yaml
│   └── summarizer.yaml
├── prompts/
│   ├── researcher_system.txt
│   ├── researcher_task.txt
│   ├── summarizer_system.txt
│   └── summarizer_task.txt
├── schemas/
│   ├── researcher_input.json
│   ├── researcher_output.json
│   ├── summarizer_input.json
│   └── summarizer_output.json
└── mlruns/                      # NEW: MLflow experiment directory
    └── (auto-created on first run)
```

---

**Last Updated:** January 26, 2026
**MLflow Version:** 2.11.0
**Status:** ✅ Implemented and tested
