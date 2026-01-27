from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import create_model
from dotenv import load_dotenv
import json
import os
import uuid
from agent_core import run_agent
from config_loader import get_config_loader
import logging

# Try to import MLflow (optional)
try:
    import mlflow
    from mlflow_utils import setup_mlflow_tracking, get_run_url
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Load environment variables from .env file
load_dotenv()

# Initialize MLflow tracking if available
if MLFLOW_AVAILABLE:
    setup_mlflow_tracking(experiment_name="AgentRuns")

logger = logging.getLogger(__name__)

app = FastAPI()


def get_agent_input_model(agent_name: str):
    """Dynamically create a Pydantic model based on agent's input schema"""
    config_path = f"configs/{agent_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Agent config '{agent_name}' not found")

    config_loader = get_config_loader()
    config = config_loader.load_config(agent_name)

    # Load input schema (supports local or mlflow:// URIs)
    schema_ref = config.get("input_schema") if isinstance(config, dict) else None

    if schema_ref is None:
        raise FileNotFoundError(f"Input schema not found for agent '{agent_name}'. Path: {schema_ref}")

    if isinstance(schema_ref, dict):
        schema = schema_ref
    else:
        schema = config_loader.load_schema(schema_ref)
    
    # Ensure schema is a dict
    if not isinstance(schema, dict):
        raise ValueError(f"Input schema must be a JSON object (dict), got {type(schema).__name__}: {schema}")
    
    # Create dynamic Pydantic model from schema
    fields = {}
    for field_name, field_info in schema.items():
        # Map JSON schema types to Python types
        field_type = str  # default to string
        if isinstance(field_info, dict) and field_info.get("type") == "integer":
            field_type = int
        elif isinstance(field_info, dict) and field_info.get("type") == "boolean":
            field_type = bool
        elif isinstance(field_info, dict) and field_info.get("type") == "number":
            field_type = float
        
        # Get description if available
        description = field_info.get("description", "") if isinstance(field_info, dict) else ""
        fields[field_name] = (field_type, description)
    
    # Create the model
    return create_model(f"{agent_name.capitalize()}Input", **fields)


@app.get("/")
async def root():
    return {"message": "Welcome to the Agentic App", "docs": "/docs"}


"""
Enhancement: Add MLflow run URL to HTTP response header.

1. If MLflow is active and run ID is available, build a full UI URL:
   e.g. http://localhost:5000/#/experiments/0/runs/<run_id>

2. Add header to response:
   response.headers["X-MLflow-Run-URL"] = <url>

3. Optional: if MLflow is inactive, skip it gracefully
"""
@app.post("/run/{agent_name}")
async def run(
    agent_name: str,
    input_data: dict = Body(..., example={"text": "Your input here"})
):
    """Run an agent with MLflow tracking"""
    # Generate request ID for traceability
    request_id = str(uuid.uuid4())
    success = False
    
    try:
        # Strip whitespace from agent_name
        agent_name = agent_name.strip()
        # input_data is provided via request body
        
        # Start MLflow run for this API request if available
        if MLFLOW_AVAILABLE and mlflow is not None:
            context = mlflow.start_run(run_name=f"{agent_name}_api_{request_id[:8]}")
        else:
            from contextlib import contextmanager
            @contextmanager
            def dummy_context():
                yield
            context = dummy_context()
        
        with context:
            # Log API request context if MLflow available
            if MLFLOW_AVAILABLE and mlflow is not None:
                mlflow.set_tag("entry_point", "rest_api")
                mlflow.set_tag("agent_name", agent_name)
                mlflow.set_tag("request_id", request_id)
                mlflow.log_param("agent_name", agent_name)
                
                # Log input info
                import hashlib
                input_hash = hashlib.sha256(json.dumps(input_data, sort_keys=True).encode()).hexdigest()
                mlflow.log_param("input_hash", input_hash)
                mlflow.log_metric("input_field_count", len(input_data))
            
            # Get the dynamic input model for this agent
            InputModel = get_agent_input_model(agent_name)
            
            # Validate input against the model
            validated_input = InputModel(**input_data).dict()
            
            # Run agent with validated input
            result = run_agent(agent_name, validated_input)
            success = True
            
            # Get current run info if MLflow available
            response_dict = result.copy() if isinstance(result, dict) else {"result": result}
            if MLFLOW_AVAILABLE and mlflow is not None:
                current_run = mlflow.active_run()
                run_id = current_run.info.run_id if current_run else request_id
                response_dict["_mlflow_run_id"] = run_id
            else:
                response_dict["_request_id"] = request_id
            
            response = JSONResponse(content=response_dict)
            if MLFLOW_AVAILABLE and mlflow is not None:
                run_url = get_run_url(run_id=run_id)
                if success and run_url:
                    response.headers["X-MLflow-Run-URL"] = run_url
            return response
    
    except FileNotFoundError as e:
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.log_param("error_type", "FileNotFoundError")
            mlflow.set_tag("status", "error")
        logger.error(f"FileNotFoundError: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as ve:
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.log_param("error_type", "ValueError")
            mlflow.set_tag("status", "error")
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.log_param("error_type", type(e).__name__)
            mlflow.set_tag("status", "error")
        logger.error(f"Exception: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

