#!/usr/bin/env python3
"""
Agent CLI Tool - Manage and run AI agents
"""

import argparse
import json
import os
import sys
import yaml
from pathlib import Path
from jinja2 import Template
from dotenv import load_dotenv
from agent_core import run_agent
import logging
import re

# Try to import MLflow (optional)
try:
    import mlflow
    from mlflow_utils import setup_mlflow_tracking, get_run_url, log_agent_artifacts
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)

# Load environment variables from .env file if present
load_dotenv()

# Initialize MLflow tracking if available
if MLFLOW_AVAILABLE:
    setup_mlflow_tracking(experiment_name="AgentRuns")

# Default boilerplate content for prompts and schemas
DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant specialized in {agent_type} tasks. 
Provide clear, concise, and accurate responses. When asked to provide data in JSON format, ensure valid JSON output."""

DEFAULT_TASK_PROMPT = """Please perform the following {agent_type} task on the provided input. 
Return your response as a valid JSON object matching the expected output schema."""

DEFAULT_INPUT_SCHEMA = {
    "text": {
        "type": "string",
        "description": "The main input text for the {agent_type} agent"
    }
}

DEFAULT_OUTPUT_SCHEMA = {
    "result": {
        "type": "string",
        "description": "The result of the {agent_type} operation"
    }
}

def ensure_agent_type_in_yaml(rendered_yaml, agent_type):
    """Ensure agent_type is present and non-empty in rendered YAML."""
    if not agent_type:
        return rendered_yaml

    lines = rendered_yaml.splitlines()
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*agent_type\s*:\s*)(.*)$", line)
        if match:
            value = match.group(2).strip()
            if value in ("", "null", "None", "\"\"", "''"):
                if value == "\"\"":
                    lines[index] = f"{match.group(1)}\"{agent_type}\""
                elif value == "''":
                    lines[index] = f"{match.group(1)}'{agent_type}'"
                else:
                    lines[index] = f"{match.group(1)}{agent_type}"
                newline = "\r\n" if "\r\n" in rendered_yaml else "\n"
                return newline.join(lines) + (newline if rendered_yaml.endswith(("\n", "\r\n")) else "")
            return rendered_yaml

    newline = "\r\n" if "\r\n" in rendered_yaml else "\n"
    return f"agent_type: {agent_type}{newline}{rendered_yaml}"


def ensure_dirs():
    """Create necessary directories if they don't exist"""
    Path("configs").mkdir(exist_ok=True)
    Path("prompts").mkdir(exist_ok=True)
    Path("schemas").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)


def load_template():
    """Load or create the Jinja2 template for agent config"""
    template_path = "templates/agent_base.yaml"

    if not os.path.exists(template_path):
        template_content = """agent_type: "{{ agent_type }}"

model:
  provider: "{{ model_provider }}"
  model_name: "{{ model_name }}"
  temperature: {{ temperature }}

prompts:
  system: "prompts/{{ agent_type }}_system.txt"
  task: "prompts/{{ agent_type }}_task.txt"

input_schema: "schemas/{{ agent_type }}_input.json"
output_schema: "schemas/{{ agent_type }}_output.json"

tools: {{ tools_list | tojson }}
"""
        with open(template_path, "w") as f:
            f.write(template_content)
        logger.info("Created template: %s", template_path)

    with open(template_path, "r") as f:
        return Template(f.read())


def generate_agent(args):
    """Generate agent files (prompts, schemas, config)"""
    ensure_dirs()

    agent_type = args.agent_type
    model_name = args.model_name or "gpt-3.5-turbo"
    temperature = args.temperature or 0
    model_provider = args.model_provider or "openai"
    tools = args.tools or []

    # Prepare file paths
    system_prompt_path = f"prompts/{agent_type}_system.txt"
    task_prompt_path = f"prompts/{agent_type}_task.txt"
    input_schema_path = f"schemas/{agent_type}_input.json"
    output_schema_path = f"schemas/{agent_type}_output.json"
    config_path = f"configs/{agent_type}.yaml"

    # Check for existing files (overwrite protection)
    existing_files = []
    for path in [system_prompt_path, task_prompt_path, input_schema_path, output_schema_path, config_path]:
        if os.path.exists(path) and not args.force:
            existing_files.append(path)

    if existing_files and not args.force:
        logger.warning("Agent '%s' already exists. Files found:", agent_type)
        for path in existing_files:
            logger.warning(" - %s", path)
        logger.info("Use --force to overwrite existing files.")
        return False

    # Create prompts
    system_content = DEFAULT_SYSTEM_PROMPT.format(agent_type=agent_type)
    with open(system_prompt_path, "w") as f:
        f.write(system_content)
    logger.info("Created: %s", system_prompt_path)

    task_content = DEFAULT_TASK_PROMPT.format(agent_type=agent_type)
    with open(task_prompt_path, "w") as f:
        f.write(task_content)
    logger.info("Created: %s", task_prompt_path)

    # Create schemas
    input_schema = {k: v for k, v in DEFAULT_INPUT_SCHEMA.items()}
    for key, value in input_schema.items():
        if "description" in value:
            value["description"] = value["description"].format(agent_type=agent_type)

    with open(input_schema_path, "w") as f:
        json.dump(input_schema, f, indent=2)
    logger.info("Created: %s", input_schema_path)

    output_schema = {k: v for k, v in DEFAULT_OUTPUT_SCHEMA.items()}
    for key, value in output_schema.items():
        if "description" in value:
            value["description"] = value["description"].format(agent_type=agent_type)

    with open(output_schema_path, "w") as f:
        json.dump(output_schema, f, indent=2)
    logger.info("Created: %s", output_schema_path)

    # Create config using template
    template = load_template()
    context = {
        "agent_type": agent_type,
        "model_provider": model_provider,
        "model_name": model_name,
        "temperature": temperature,
        "tools_list": tools,
    }

    rendered_yaml = template.render(**context)
    rendered_yaml = ensure_agent_type_in_yaml(rendered_yaml, agent_type)
    with open(config_path, "w") as f:
        f.write(rendered_yaml)
    logger.info("Created: %s", config_path)

    # Print summary
    logger.info("Agent '%s' generated successfully.", agent_type)
    logger.info("Config: %s", config_path)
    logger.info("Prompts: %s, %s", system_prompt_path, task_prompt_path)
    logger.info("Schemas: %s, %s", input_schema_path, output_schema_path)

    return True


def load_input(input_arg):
    """Load input from JSON string or file"""
    if os.path.isfile(input_arg):
        with open(input_arg, "r") as f:
            return json.load(f)
    else:
        return json.loads(input_arg)


def run_agent_command(args):
    """Run an agent with given input"""
    agent_type = args.agent_type

    try:
        # Start MLflow run for CLI invocation if available
        if MLFLOW_AVAILABLE and mlflow is not None:
            context = mlflow.start_run(run_name=f"{agent_type}_cli")
        else:
            from contextlib import contextmanager
            @contextmanager
            def dummy_context():
                yield
            context = dummy_context()

        with context:
            # Log CLI context if MLflow available
            if MLFLOW_AVAILABLE and mlflow is not None:
                mlflow.set_tag("entry_point", "cli")
                mlflow.set_tag("agent_name", agent_type)
                mlflow.log_param("agent_name", agent_type)

            # Load input
            user_input = load_input(args.input)

            # Log input info if MLflow available
            if MLFLOW_AVAILABLE and mlflow is not None:
                import hashlib
                input_hash = hashlib.sha256(json.dumps(user_input, sort_keys=True).encode()).hexdigest()
                mlflow.log_param("input_hash", input_hash)
                mlflow.log_metric("input_field_count", len(user_input))

            # Run agent
            result = run_agent(agent_type, user_input)

            # Log result
            logger.info("Agent execution successful.")
            logger.info("Result:\n%s", json.dumps(result, indent=2))

            # Log MLflow run info if available
            if MLFLOW_AVAILABLE and mlflow is not None:
                current_run = mlflow.active_run()
                run_id = current_run.info.run_id if current_run else "unknown"
                logger.info("MLflow Run ID: %s", run_id)
                run_url = get_run_url(run_id=run_id) if MLFLOW_AVAILABLE else None
                if run_url:
                    logger.info("View in MLflow UI: %s", run_url)

            return True

    except FileNotFoundError as e:
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.log_param("error_type", "FileNotFoundError")
            mlflow.set_tag("status", "error")
        logger.error("Error: %s", e)
        logger.error("Agent '%s' config not found. Run 'generate' first.", agent_type)
        return False
    except json.JSONDecodeError as e:
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.log_param("error_type", "JSONDecodeError")
            mlflow.set_tag("status", "error")
        logger.error("Invalid JSON input: %s", e)
        return False
    except ValueError as e:
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.log_param("error_type", "ValueError")
            mlflow.set_tag("status", "error")
        logger.error("Error: %s", e)
        return False
    except Exception as e:
        if MLFLOW_AVAILABLE and mlflow is not None:
            mlflow.log_param("error_type", type(e).__name__)
            mlflow.set_tag("status", "error")
        logger.error("Error: %s", e)
        return False


def log_artifacts_command(args):
    """Log local prompts/schemas/config to MLflow and optionally update config."""
    if not MLFLOW_AVAILABLE or mlflow is None:
        logger.error("MLflow not available. Install mlflow to log artifacts.")
        return False

    agent_type = args.agent_type
    config_path = args.config_path or f"configs/{agent_type}.yaml"

    if not os.path.exists(config_path):
        logger.error("Config not found at %s", config_path)
        return False

    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    prompts = config.get("prompts", {}) if isinstance(config, dict) else {}
    system_prompt = args.system_path or prompts.get("system") or f"prompts/{agent_type}_system.txt"
    task_prompt = args.task_path or prompts.get("task") or f"prompts/{agent_type}_task.txt"

    input_schema = args.input_schema or config.get("input_schema") or f"schemas/{agent_type}_input.json"
    output_schema = args.output_schema or config.get("output_schema") or f"schemas/{agent_type}_output.json"

    # If schema references are MLflow URIs, fall back to local defaults
    if isinstance(input_schema, str) and input_schema.startswith("mlflow://"):
        input_schema = f"schemas/{agent_type}_input.json"
    if isinstance(output_schema, str) and output_schema.startswith("mlflow://"):
        output_schema = f"schemas/{agent_type}_output.json"
    if isinstance(system_prompt, str) and system_prompt.startswith("mlflow://"):
        system_prompt = f"prompts/{agent_type}_system.txt"
    if isinstance(task_prompt, str) and task_prompt.startswith("mlflow://"):
        task_prompt = f"prompts/{agent_type}_task.txt"

    try:
        logged = log_agent_artifacts(
            agent_name=agent_type,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            input_schema=input_schema,
            output_schema=output_schema,
            config_file=config_path,
            version=args.version,
            tracking_uri=args.tracking_uri,
            experiment_name=args.experiment,
            run_name=args.run_name,
            description=args.description,
        )
    except Exception as e:
        logger.error("Failed to log artifacts: %s", e)
        return False

    logger.info("Logged artifacts for '%s'. Run ID: %s", agent_type, logged.get("run_id"))
    for key, uri in logged.get("uris", {}).items():
        logger.info("%s: %s", key, uri)
    if MLFLOW_AVAILABLE and mlflow is not None and logged.get("run_id"):
        run_url = get_run_url(run_id=logged.get("run_id"))
        if run_url:
            logger.info("MLflow Run URL: %s", run_url)

    if args.update_config:
        config = config if isinstance(config, dict) else {}
        config.setdefault("prompts", {})
        fallback_paths = {}
        if "system_prompt" in logged.get("uris", {}):
            config["prompts"]["system"] = logged["uris"]["system_prompt"]
            if isinstance(system_prompt, str) and os.path.exists(system_prompt):
                fallback_paths[logged["uris"]["system_prompt"]] = system_prompt
        if "task_prompt" in logged.get("uris", {}):
            config["prompts"]["task"] = logged["uris"]["task_prompt"]
            if isinstance(task_prompt, str) and os.path.exists(task_prompt):
                fallback_paths[logged["uris"]["task_prompt"]] = task_prompt
        if "input_schema" in logged.get("uris", {}):
            config["input_schema"] = logged["uris"]["input_schema"]
            if isinstance(input_schema, str) and os.path.exists(input_schema):
                fallback_paths[logged["uris"]["input_schema"]] = input_schema
        if "output_schema" in logged.get("uris", {}):
            config["output_schema"] = logged["uris"]["output_schema"]
            if isinstance(output_schema, str) and os.path.exists(output_schema):
                fallback_paths[logged["uris"]["output_schema"]] = output_schema

        mlflow_config = config.get("mlflow", {}) if isinstance(config, dict) else {}
        mlflow_config["tracking_uri"] = args.tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
        if logged.get("run_id"):
            mlflow_config["run_id"] = logged["run_id"]
        if fallback_paths:
            mlflow_config["fallback_paths"] = fallback_paths
        config["mlflow"] = mlflow_config

        with open(config_path, "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        logger.info("Updated config with MLflow URIs: %s", config_path)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Agent CLI - Manage and run AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate a new agent
  python agent_cli.py generate --agent_type summarizer
  
  # Generate with custom model
  python agent_cli.py generate --agent_type analyzer --model_name gpt-4
  
  # Run agent with JSON input
  python agent_cli.py run --agent_type summarizer --input '{"text": "Hello world"}'
  
  # Run agent with input from file
  python agent_cli.py run --agent_type summarizer --input input.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate subcommand
    gen_parser = subparsers.add_parser("generate", help="Generate a new agent")
    gen_parser.add_argument("--agent_type", required=True, help="Type/name of the agent")
    gen_parser.add_argument("--model_name", default="gpt-3.5-turbo", help="OpenAI model to use")
    gen_parser.add_argument("--model_provider", default="openai", help="Model provider")
    gen_parser.add_argument("--temperature", type=float, default=0, help="Model temperature (0-1)")
    gen_parser.add_argument("--tools", nargs="*", default=[], help="Tools available to agent")
    gen_parser.add_argument("--force", action="store_true", help="Overwrite existing files")
    gen_parser.set_defaults(func=generate_agent)
    
    # Run subcommand
    run_parser = subparsers.add_parser("run", help="Run an agent")
    run_parser.add_argument("--agent_type", required=True, help="Type/name of the agent to run")
    run_parser.add_argument("--input", required=True, help="JSON input (string or file path)")
    run_parser.set_defaults(func=run_agent_command)

    # Log artifacts subcommand
    log_parser = subparsers.add_parser("log-artifacts", help="Log prompts/schemas/config to MLflow")
    log_parser.add_argument("--agent_type", required=True, help="Type/name of the agent")
    log_parser.add_argument("--version", default="v1", help="Asset version label (e.g., v1, v2)")
    log_parser.add_argument("--experiment", default="AgentAssets", help="MLflow experiment name")
    log_parser.add_argument("--tracking_uri", default=None, help="Override MLFLOW_TRACKING_URI")
    log_parser.add_argument("--run_name", default=None, help="Optional MLflow run name")
    log_parser.add_argument("--description", default=None, help="Optional description tag")
    log_parser.add_argument("--config_path", default=None, help="Path to agent config YAML")
    log_parser.add_argument("--system_path", default=None, help="Override system prompt path")
    log_parser.add_argument("--task_path", default=None, help="Override task prompt path")
    log_parser.add_argument("--input_schema", default=None, help="Override input schema path")
    log_parser.add_argument("--output_schema", default=None, help="Override output schema path")
    log_parser.add_argument("--update_config", action="store_true", help="Rewrite config with mlflow:// URIs")
    log_parser.set_defaults(func=log_artifacts_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    success = args.func(args)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
