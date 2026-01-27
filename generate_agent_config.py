import yaml
from jinja2 import Template
import argparse
import os
import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def generate_config(template_path, output_path, context):
    with open(template_path, "r") as f:
        template = Template(f.read())
    rendered_yaml = template.render(**context)
    with open(output_path, "w") as f:
        f.write(rendered_yaml)
    logger.info("Generated: %s", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_type", required=True)
    parser.add_argument("--model_provider", default="openai")
    parser.add_argument("--model_name", default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--prompt_system", required=True)
    parser.add_argument("--prompt_task", required=True)
    parser.add_argument("--input_schema", required=True)
    parser.add_argument("--output_schema", required=True)
    parser.add_argument("--tools", nargs="*", default=[])

    args = parser.parse_args()
    context = {
        "agent_type": args.agent_type,
        "model_provider": args.model_provider,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "prompt_system": args.prompt_system,
        "prompt_task": args.prompt_task,
        "input_schema": args.input_schema,
        "output_schema": args.output_schema,
        "tools_list": args.tools,
    }

    os.makedirs("configs", exist_ok=True)
    generate_config("templates/agent_base.yaml", f"configs/{args.agent_type}.yaml", context)
