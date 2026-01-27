"""
MLflow Utilities - Helper functions for experiment tracking and run management
"""

import os
import logging
import traceback as tb
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union

try:
    import mlflow
    from mlflow import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    MlflowClient = None

logger = logging.getLogger(__name__)


def setup_mlflow_tracking(
    tracking_uri: Optional[str] = None,
    experiment_name: str = "AgentRuns",
    registry_uri: Optional[str] = None
) -> str:
    """
    Configure MLflow tracking for the application
    
    Args:
        tracking_uri: MLflow tracking server URI (default: from env or sqlite:///mlflow.db)
        experiment_name: Name of the experiment to use
        
    Returns:
        Experiment ID
    """
    if not MLFLOW_AVAILABLE:
        logger.warning("MLflow not available. Run `pip install mlflow` to enable experiment tracking.")
        return "mlflow_not_available"
    
    # Set tracking URI
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)

    # Set registry URI if provided
    registry_uri = registry_uri or os.getenv("MLFLOW_REGISTRY_URI")
    if registry_uri:
        mlflow.set_registry_uri(registry_uri)
    
    logger.info(f"MLflow tracking configured: {tracking_uri}")
    
    # Set experiment
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        logger.info(f"Created MLflow experiment '{experiment_name}' (ID: {experiment_id})")
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"Using existing MLflow experiment '{experiment_name}' (ID: {experiment_id})")
    
    return experiment_id


def log_agent_artifacts(
    agent_name: str,
    system_prompt: Optional[Union[str, Dict[str, Any]]] = None,
    task_prompt: Optional[Union[str, Dict[str, Any]]] = None,
    input_schema: Optional[Union[str, Dict[str, Any]]] = None,
    output_schema: Optional[Union[str, Dict[str, Any]]] = None,
    config_file: Optional[Union[str, Dict[str, Any]]] = None,
    version: str = "v1",
    tracking_uri: Optional[str] = None,
    experiment_name: str = "AgentAssets",
    run_name: Optional[str] = None,
    description: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Log agent prompts, schemas, and config as MLflow artifacts.

    Returns:
        Dict with run_id and logical mlflow:// URIs for logged assets.
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        raise RuntimeError("MLflow not available. Install mlflow to log artifacts.")

    setup_mlflow_tracking(tracking_uri=tracking_uri, experiment_name=experiment_name)

    def read_text(source: Union[str, Dict[str, Any]]) -> str:
        if isinstance(source, dict):
            return json.dumps(source, indent=2)
        if isinstance(source, str):
            path = Path(source)
            if path.exists():
                return path.read_text(encoding="utf-8")
            return source
        raise TypeError(f"Unsupported artifact source type: {type(source).__name__}")

    run_name = run_name or f"{agent_name}_assets_{version}"
    logged = {
        "agent_name": agent_name,
        "version": version,
        "uris": {},
    }

    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        run_id = run.info.run_id if run else ""
        logged["run_id"] = run_id

        mlflow.set_tag("asset_type", "agent_assets")
        mlflow.set_tag("agent_name", agent_name)
        mlflow.set_tag("asset_version", version)
        if description:
            mlflow.set_tag("description", description)
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        if system_prompt is not None:
            content = read_text(system_prompt)
            artifact_file = f"prompts/{agent_name}/system_{version}.txt"
            mlflow.log_text(content, artifact_file)
            logged["uris"]["system_prompt"] = f"mlflow://prompts/{agent_name}/system_{version}"
            mlflow.set_tag("system_prompt_name", f"{agent_name}-system-{version}")

        if task_prompt is not None:
            content = read_text(task_prompt)
            artifact_file = f"prompts/{agent_name}/task_{version}.txt"
            mlflow.log_text(content, artifact_file)
            logged["uris"]["task_prompt"] = f"mlflow://prompts/{agent_name}/task_{version}"
            mlflow.set_tag("task_prompt_name", f"{agent_name}-task-{version}")

        if input_schema is not None:
            content = read_text(input_schema)
            artifact_file = f"schemas/{agent_name}/input_{version}.json"
            mlflow.log_text(content, artifact_file)
            logged["uris"]["input_schema"] = f"mlflow://schemas/{agent_name}/input_{version}"
            mlflow.set_tag("input_schema_name", f"{agent_name}-input-{version}")

        if output_schema is not None:
            content = read_text(output_schema)
            artifact_file = f"schemas/{agent_name}/output_{version}.json"
            mlflow.log_text(content, artifact_file)
            logged["uris"]["output_schema"] = f"mlflow://schemas/{agent_name}/output_{version}"
            mlflow.set_tag("output_schema_name", f"{agent_name}-output-{version}")

        if config_file is not None:
            content = read_text(config_file)
            artifact_file = f"configs/{agent_name}/config_{version}.yaml"
            mlflow.log_text(content, artifact_file)
            logged["uris"]["config"] = f"mlflow://configs/{agent_name}/config_{version}"
            mlflow.set_tag("config_name", f"{agent_name}-config-{version}")

    return logged


def start_agent_run(
    agent_name: str,
    entry_point: str = "cli",
    input_data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Start a new MLflow run for agent execution
    
    Args:
        agent_name: Name of the agent
        entry_point: How the agent was invoked ("cli" or "api")
        input_data: Input data passed to the agent
        
    Returns:
        Run ID
    """
    # Generate unique run name
    run_name = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    # Start run
    with mlflow.start_run(run_name=run_name):
        run = mlflow.active_run()
        run_id = run.info.run_id
        
        # Log tags
        mlflow.set_tag("entry_point", entry_point)
        mlflow.set_tag("agent_name", agent_name)
        mlflow.set_tag("execution_timestamp", datetime.now().isoformat())
        
        # Log params
        mlflow.log_param("agent_name", agent_name)
        
        if input_data:
            # Log input hash for privacy
            import hashlib
            import json
            input_hash = hashlib.sha256(
                json.dumps(input_data, sort_keys=True).encode()
            ).hexdigest()
            mlflow.log_param("input_hash", input_hash)
            mlflow.log_metric("input_field_count", len(input_data))
        
        logger.info(f"Started MLflow run '{run_name}' (ID: {run_id})")
        
        return run_id


def log_openai_call_params(
    model_name: str,
    temperature: float,
    system_prompt: str,
    task_prompt: str,
    user_input: str
):
    """
    Log parameters for OpenAI API call
    
    Args:
        model_name: Model name (e.g., "gpt-3.5-turbo")
        temperature: Temperature parameter
        system_prompt: System prompt text
        task_prompt: Task prompt text
        user_input: User input text
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        return
    
    mlflow.log_param("model_name", model_name)
    mlflow.log_param("temperature", temperature)
    
    # Estimate tokens (rough approximation: 1 token ≈ 4 characters)
    mlflow.log_metric("system_prompt_chars", len(system_prompt))
    mlflow.log_metric("task_prompt_chars", len(task_prompt))
    mlflow.log_metric("user_input_chars", len(user_input))


def log_openai_response(
    response_obj: Any,
    response_text: str,
    api_latency_ms: float
):
    """
    Log OpenAI API response metrics
    
    Args:
        response_obj: OpenAI response object (contains usage info)
        response_text: Response content text
        api_latency_ms: API call latency in milliseconds
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        return
    
    if hasattr(response_obj, 'usage'):
        mlflow.log_metric("tokens_prompt", response_obj.usage.prompt_tokens)
        mlflow.log_metric("tokens_completion", response_obj.usage.completion_tokens)
        mlflow.log_metric("tokens_total", response_obj.usage.total_tokens)
    
    mlflow.log_metric("openai_api_latency_ms", api_latency_ms)
    mlflow.log_metric("response_length_chars", len(response_text))


def log_json_parsing(
    parse_stage: str,
    parse_attempts: int,
    parsed_json: Dict[str, Any]
):
    """
    Log JSON parsing stage and metrics
    
    Args:
        parse_stage: Parsing stage reached ("original", "retry", or "repair")
        parse_attempts: Number of attempts needed (1, 2, or 3)
        parsed_json: Successfully parsed JSON
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        return
    
    mlflow.set_tag("parse_stage", parse_stage)
    mlflow.log_metric("parse_attempts_needed", parse_attempts)
    
    if parse_stage != "original":
        mlflow.log_metric("parse_recovery_required", 1)
    
    mlflow.log_param("parse_final_stage", parse_stage)
    mlflow.log_metric("parsed_json_field_count", len(parsed_json))
    
    import json
    mlflow.log_metric("parsed_json_size_bytes", len(json.dumps(parsed_json)))


def log_output_validation(
    output_json: Dict[str, Any],
    validation_errors: Optional[list] = None
):
    """
    Log output schema validation metrics
    
    Args:
        output_json: Output JSON after validation
        validation_errors: List of validation errors (if any)
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        return
    
    import json
    mlflow.log_metric("output_field_count", len(output_json))
    mlflow.log_metric("output_size_bytes", len(json.dumps(output_json)))
    mlflow.log_param("output_fields", list(output_json.keys()))
    
    if validation_errors:
        mlflow.log_metric("validation_errors_count", len(validation_errors))


def log_agent_output(
    output_json: Dict[str, Any],
    artifact_file: str = "output.json"
):
    """
    Log agent output JSON as an MLflow artifact.

    Args:
        output_json: Output JSON data
        artifact_file: Artifact filename to store under the run
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        return

    mlflow.log_text(json.dumps(output_json, indent=2), artifact_file)


def log_execution_success(total_latency_ms: float):
    """
    Log successful execution completion
    
    Args:
        total_latency_ms: Total execution time in milliseconds
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        return
    
    mlflow.set_tag("status", "success")
    mlflow.log_metric("total_latency_ms", total_latency_ms)


def log_execution_error(
    exception: Exception,
    total_latency_ms: Optional[float] = None,
    stage_reached: str = "unknown"
):
    """
    Log execution error with traceback
    
    Args:
        exception: The exception that occurred
        total_latency_ms: Total execution time (if available)
        stage_reached: Stage where error occurred
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        return
    
    mlflow.set_tag("status", "error")
    mlflow.set_tag("error_type", type(exception).__name__)
    mlflow.set_tag("error_stage", stage_reached)
    
    # Log error message (truncate if too long)
    error_msg = str(exception)
    mlflow.log_param("error_message", error_msg[:500])
    
    # Log full traceback as artifact
    traceback_text = tb.format_exc()
    mlflow.log_text(traceback_text, "traceback.txt")
    
    if total_latency_ms is not None:
        mlflow.log_metric("total_latency_ms", total_latency_ms)
    
    logger.error(f"Logged error to MLflow: {type(exception).__name__}: {error_msg[:100]}")


def end_agent_run(status: str = "SUCCESS"):
    """
    End the current MLflow run
    
    Args:
        status: Run status ("SUCCESS" or "FAILED")
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        return
    
    mlflow.end_run(status=status)
    logger.debug(f"Ended MLflow run with status: {status}")


def get_run_info(run_id: str) -> Dict[str, Any]:
    """
    Get information about a completed run
    
    Args:
        run_id: MLflow run ID
        
    Returns:
        Dictionary with run information
    """
    if not MLFLOW_AVAILABLE or MlflowClient is None:
        return {}
    
    client = MlflowClient()
    run = client.get_run(run_id)
    
    return {
        "run_id": run.info.run_id,
        "experiment_id": run.info.experiment_id,
        "status": run.info.status,
        "start_time": run.info.start_time,
        "end_time": run.info.end_time,
        "duration_ms": (run.info.end_time - run.info.start_time) if run.info.end_time else None,
        "params": run.data.params,
        "metrics": run.data.metrics,
        "tags": run.data.tags,
    }


def get_run_url(tracking_uri: Optional[str] = None, run_id: Optional[str] = None) -> str:
    """
    Generate URL to view run in MLflow UI
    
    Args:
        tracking_uri: MLflow tracking server URI
        run_id: MLflow run ID (default: current active run)
        
    Returns:
        URL to MLflow UI for this run
    """
    if not MLFLOW_AVAILABLE or mlflow is None:
        return ""
    
    if run_id is None:
        active_run = mlflow.active_run()
        if active_run is None:
            return ""
        run_id = active_run.info.run_id
    
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # For local tracking, construct local URL
    if tracking_uri.startswith("sqlite://") or tracking_uri.startswith("file://"):
        return f"http://localhost:5000/#/experiments/0/runs/{run_id}"
    else:
        # For remote server
        return f"{tracking_uri.rstrip('/')}/#/experiments/0/runs/{run_id}"
