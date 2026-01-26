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
from agent_core import run_agent

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
        print(f"📋 Created template: {template_path}")
    
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
        print(f"⚠️  Agent '{agent_type}' already exists. Files found:")
        for f in existing_files:
            print(f"   - {f}")
        print(f"\n💡 Use --force to overwrite existing files")
        return False
    
    # Create prompts
    system_content = DEFAULT_SYSTEM_PROMPT.format(agent_type=agent_type)
    with open(system_prompt_path, "w") as f:
        f.write(system_content)
    print(f"✅ Created: {system_prompt_path}")
    
    task_content = DEFAULT_TASK_PROMPT.format(agent_type=agent_type)
    with open(task_prompt_path, "w") as f:
        f.write(task_content)
    print(f"✅ Created: {task_prompt_path}")
    
    # Create schemas
    input_schema = {k: v for k, v in DEFAULT_INPUT_SCHEMA.items()}
    for key, value in input_schema.items():
        if "description" in value:
            value["description"] = value["description"].format(agent_type=agent_type)
    
    with open(input_schema_path, "w") as f:
        json.dump(input_schema, f, indent=2)
    print(f"✅ Created: {input_schema_path}")
    
    output_schema = {k: v for k, v in DEFAULT_OUTPUT_SCHEMA.items()}
    for key, value in output_schema.items():
        if "description" in value:
            value["description"] = value["description"].format(agent_type=agent_type)
    
    with open(output_schema_path, "w") as f:
        json.dump(output_schema, f, indent=2)
    print(f"✅ Created: {output_schema_path}")
    
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
    with open(config_path, "w") as f:
        f.write(rendered_yaml)
    print(f"✅ Created: {config_path}")
    
    # Print summary
    print(f"\n📦 Agent '{agent_type}' generated successfully!")
    print(f"   Config: {config_path}")
    print(f"   Prompts: {system_prompt_path}, {task_prompt_path}")
    print(f"   Schemas: {input_schema_path}, {output_schema_path}")
    
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
        # Load input
        user_input = load_input(args.input)
        
        # Run agent
        result = run_agent(agent_type, user_input)
        
        # Print result
        print("\n✅ Agent execution successful!")
        print(json.dumps(result, indent=2))
        
        return True
    
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"   Agent '{agent_type}' config not found. Run 'generate' first.")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON input - {e}")
        return False
    except ValueError as e:
        print(f"❌ Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="🤖 Agent CLI - Manage and run AI agents",
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
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    success = args.func(args)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
