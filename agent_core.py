import os
import json
import time
import logging
import hashlib
from openai import OpenAI
from config_loader import get_config_loader

logger = logging.getLogger(__name__)


def load_agent_config(agent_name: str):
    config_loader = get_config_loader()
    return config_loader.load_config(agent_name)


def load_prompt(path: str):
    """Load prompt from file or MLflow artifact"""
    config_loader = get_config_loader()
    try:
        return config_loader.load_prompt(path)
    except Exception:
        # Fallback to direct file loading for backward compatibility
        if os.path.exists(path):
            with open(path, "r") as f:
                return f.read()
        raise


def call_openai(
    model_name,
    system_prompt,
    task_prompt,
    user_input,
    temperature=0,
    log_response=None
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the agent.")
    
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt + "\n\n" + user_input}
    ]
    
    # Record start time for latency tracking
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    
    # Calculate latency
    latency_ms = (time.time() - start_time) * 1000
    
    # Log OpenAI response metrics to MLflow if provided
    if log_response:
        log_response(response, response.choices[0].message.content, latency_ms)
    
    return response, response.choices[0].message.content


def try_parse_json_with_retry(
    llm_response: str,
    model_name: str,
    system_prompt: str,
    task_prompt: str,
    user_input: str,
    temperature: float,
    log_func=None,
    log_response=None
) -> tuple[dict, str]:
    """
    Try to parse LLM response as JSON with retry and repair mechanism.

    Returns:
        tuple: (parsed_json, stage_description)
        - stage_description: one of "original", "retry", "repair"

    Raises:
        ValueError: If all attempts fail
    """

    # Stage 1: Try original response
    logger.info("JSON parsing stage: original")
    try:
        parsed = json.loads(llm_response)
        logger.info("JSON parsing succeeded at stage: original")
        if log_func:
            log_func("original", 1, parsed)
        return parsed, "original"
    except json.JSONDecodeError as e:
        logger.warning("JSON parsing failed at stage: original: %s", e)

    # Stage 2: Retry - call LLM again with same prompt
    logger.info("JSON parsing stage: retry")
    try:
        retry_response_obj, retry_response = call_openai(
            model_name=model_name,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            user_input=user_input,
            temperature=temperature,
            log_response=log_response
        )
        parsed = json.loads(retry_response)
        logger.info("JSON parsing succeeded at stage: retry")
        if log_func:
            log_func("retry", 2, parsed)
        return parsed, "retry"
    except json.JSONDecodeError as e:
        logger.warning("JSON parsing failed at stage: retry: %s", e)

    # Stage 3: Repair - ask LLM to fix the broken JSON
    logger.info("JSON parsing stage: repair")
    repair_task = f"""The following response was not valid JSON. Please fix it to be valid JSON matching the expected structure.

Broken response:
{llm_response}

Return ONLY the fixed JSON object, nothing else."""

    try:
        repair_response_obj, repaired_response = call_openai(
            model_name=model_name,
            system_prompt=system_prompt,
            task_prompt=repair_task,
            user_input="",  # Empty since repair task is self-contained
            temperature=temperature,
            log_response=log_response
        )
        parsed = json.loads(repaired_response)
        logger.info("JSON parsing succeeded at stage: repair")
        if log_func:
            log_func("repair", 3, parsed)
        return parsed, "repair"
    except json.JSONDecodeError as e:
        logger.error("JSON parsing failed at stage: repair: %s", e)
        raise ValueError(
            "Failed to get valid JSON after 3 attempts (original, retry, repair). "
            f"Last response: {repaired_response[:200]}"
        )


def run_agent(agent_name: str, user_input: dict):
    """Run an agent with MLflow experiment tracking"""
    # Import MLflow utilities lazily (to avoid import errors if MLflow not installed)
    mlflow_available = False
    log_funcs = {}

    try:
        import mlflow
        from mlflow_utils import (
            setup_mlflow_tracking,
            log_openai_call_params,
            log_openai_response,
            log_json_parsing,
            log_output_validation,
            log_agent_output,
            log_execution_success,
            log_execution_error,
            end_agent_run,
            MLFLOW_AVAILABLE,
        )
        mlflow_available = MLFLOW_AVAILABLE
        log_funcs = {
            "call_params": log_openai_call_params,
            "response": log_openai_response,
            "parsing": log_json_parsing,
            "validation": log_output_validation,
            "output": log_agent_output,
            "success": log_execution_success,
            "error": log_execution_error,
            "end_run": end_agent_run,
        }
        if mlflow_available and mlflow is not None:
            setup_mlflow_tracking(experiment_name="AgentRuns")
    except ImportError:
        mlflow = None
        mlflow_available = False

    # Execution start time for latency tracking
    execution_start_time = time.time()
    local_run_id = f"local-{int(execution_start_time * 1000)}"
    input_payload = None
    try:
        input_payload = json.dumps(user_input, sort_keys=True, default=str)
    except (TypeError, ValueError) as e:
        logger.warning("Failed to JSON-serialize input for hashing: %s", e)
        input_payload = repr(user_input)
    input_hash = hashlib.sha256(input_payload.encode("utf-8")).hexdigest()

    # Start MLflow run if available and not already active
    owns_run = False
    run_id = None
    if mlflow_available and mlflow is not None:
        active_run = mlflow.active_run()
        if active_run is None:
            run_context = mlflow.start_run(run_name=f"{agent_name}_{int(time.time() * 1000) % 10000}")
            owns_run = True
            run_id = run_context.info.run_id
        else:
            from contextlib import contextmanager
            @contextmanager
            def dummy_context():
                yield
            run_context = dummy_context()
            run_id = active_run.info.run_id
    else:
        # Dummy context manager for when MLflow is not available
        from contextlib import contextmanager
        @contextmanager
        def dummy_context():
            yield
        run_context = dummy_context()
        run_id = None

    if not run_id:
        run_id = local_run_id

    logger.info(
        "Agent run started: agent=%s run_id=%s input_hash=%s",
        agent_name,
        run_id,
        input_hash,
    )

    with run_context:
        try:
            # Log entry point and basic params (only if MLflow available)
            if mlflow_available and mlflow is not None and owns_run:
                mlflow.set_tag("entry_point", "agent_core")
                mlflow.set_tag("agent_name", agent_name)
                mlflow.log_param("agent_name", agent_name)

                # Log input info
                if input_hash:
                    mlflow.log_param("input_hash", input_hash)
                mlflow.log_metric("input_field_count", len(user_input))

            config = load_agent_config(agent_name)

            # Load prompts using config_loader for MLflow support
            system_prompt = load_prompt(config["prompts"]["system"])
            task_prompt = load_prompt(config["prompts"]["task"])

            # Load input schema
            input_schema_path = config["input_schema"]
            config_loader = get_config_loader()
            if isinstance(input_schema_path, dict):
                input_schema = input_schema_path
            else:
                try:
                    input_schema = config_loader.load_schema(input_schema_path)
                except:
                    # Fallback to direct file loading
                    with open(input_schema_path, "r") as f:
                        input_schema = json.load(f)

            # Currently using only one input field (e.g., 'text')
            if not isinstance(input_schema, dict):
                raise TypeError(f"Expected input_schema to be dict, got {type(input_schema).__name__}: {repr(input_schema)}")

            input_key = list(input_schema.keys())[0]
            input_value = user_input.get(input_key)

            if not input_value:
                raise ValueError(f"Missing input field: '{input_key}'")

            # Call the LLM
            model_name = config["model"]["model_name"]
            temperature = config["model"].get("temperature", 0)

            # Log OpenAI call parameters
            if mlflow_available and log_funcs:
                log_funcs["call_params"](model_name, temperature, system_prompt, task_prompt, input_value)

            initial_response_obj, initial_response = call_openai(
                model_name=model_name,
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                user_input=input_value,
                temperature=temperature,
                log_response=log_funcs.get("response") if mlflow_available else None
            )

            # Try to parse LLM output with retry and repair mechanism
            parsed, stage = try_parse_json_with_retry(
                llm_response=initial_response,
                model_name=model_name,
                system_prompt=system_prompt,
                task_prompt=task_prompt,
                user_input=input_value,
                temperature=temperature,
                log_func=log_funcs.get("parsing") if mlflow_available else None,
                log_response=log_funcs.get("response") if mlflow_available else None
            )

            logger.info("JSON parsing succeeded at stage: %s", stage)

            # Load and validate against output schema from config
            output_schema_path = config.get("output_schema")
            output_schema = {}

            if output_schema_path:
                if isinstance(output_schema_path, dict):
                    output_schema = output_schema_path
                else:
                    try:
                        output_schema = config_loader.load_schema(output_schema_path)
                    except:
                        # Fallback to direct file loading
                        if os.path.exists(output_schema_path):
                            with open(output_schema_path, "r") as f:
                                output_schema = json.load(f)

            # Validate output fields if schema exists
            validation_errors = []
            if output_schema and isinstance(output_schema, dict):
                for field, field_info in output_schema.items():
                    if field not in parsed:
                        raise ValueError(f"Missing field in output: '{field}'")

                    value = parsed[field]
                    expected_type = field_info.get("type") if isinstance(field_info, dict) else "string"

                    if expected_type == "string" and not isinstance(value, str):
                        validation_errors.append(f"Field '{field}' must be a string.")
                    elif expected_type == "integer" and not isinstance(value, int):
                        validation_errors.append(f"Field '{field}' must be an integer.")
                    elif expected_type.startswith("list[") and not isinstance(value, list):
                        validation_errors.append(f"Field '{field}' must be a list.")
                    # Optional: check list element types
                    if expected_type == "list[string]":
                        if not all(isinstance(item, str) for item in value):
                            validation_errors.append(f"All items in '{field}' must be strings.")

                if validation_errors:
                    raise ValueError("; ".join(validation_errors))

            # Log validation and output metrics
            if mlflow_available and log_funcs:
                log_funcs["validation"](parsed, validation_errors if validation_errors else None)
                log_funcs["output"](parsed)

            # Log successful completion
            total_latency_ms = (time.time() - execution_start_time) * 1000
            if mlflow_available and log_funcs:
                log_funcs["success"](total_latency_ms)

            logger.info(
                "Agent run completed: agent=%s run_id=%s status=success latency_ms=%.1f",
                agent_name,
                run_id,
                total_latency_ms,
            )

            if mlflow_available and log_funcs and owns_run:
                log_funcs["end_run"](status="SUCCESS")
            return parsed

        except Exception as e:
            # Log error to MLflow
            total_latency_ms = (time.time() - execution_start_time) * 1000

            error_stage = "unknown"
            if "config" in locals():
                error_stage = "config_loaded"
            if "parsed" in locals():
                error_stage = "json_parsing"
            if "validation_errors" in locals():
                error_stage = "validation"

            if mlflow_available and log_funcs:
                log_funcs["error"](e, total_latency_ms, error_stage)
                if owns_run:
                    log_funcs["end_run"](status="FAILED")

            logger.exception(
                "Agent run failed: agent=%s run_id=%s stage=%s latency_ms=%.1f",
                agent_name,
                run_id,
                error_stage,
                total_latency_ms,
            )
            raise

