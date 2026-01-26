from fastapi import FastAPI, HTTPException
from pydantic import create_model
from dotenv import load_dotenv
import json
import os
import yaml
from agent_core import run_agent

# Load environment variables from .env file
load_dotenv()

app = FastAPI()


def get_agent_input_model(agent_name: str):
    """Dynamically create a Pydantic model based on agent's input schema"""
    config_path = f"configs/{agent_name}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Agent config '{agent_name}' not found")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Load input schema
    schema_path = config.get("input_schema")
    print(f"DEBUG: schema_path = {repr(schema_path)}, type = {type(schema_path)}")
    
    if not schema_path or not os.path.exists(schema_path):
        raise FileNotFoundError(f"Input schema not found for agent '{agent_name}'. Path: {schema_path}")
    
    with open(schema_path, "r") as f:
        raw_content = f.read()
        print(f"DEBUG: raw file content = {repr(raw_content[:100])}")
        schema = json.loads(raw_content)
    
    print(f"DEBUG: schema type after json.load = {type(schema)}, schema = {schema}")
    
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


@app.post("/run/{agent_name}")
async def run(agent_name: str, input_data: dict):
    try:
        # Strip whitespace from agent_name
        agent_name = agent_name.strip()
        
        # Get the dynamic input model for this agent
        InputModel = get_agent_input_model(agent_name)
        
        # Validate input against the model
        validated_input = InputModel(**input_data).dict()
        
        # Run agent with validated input
        result = run_agent(agent_name, validated_input)
        return result
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
