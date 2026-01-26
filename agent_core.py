import os
import yaml
import json
from openai import OpenAI

def load_agent_config(agent_name: str):
    config_path = f"configs/{agent_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config found for agent '{agent_name}'")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_prompt(path: str):
    with open(path, "r") as f:
        return f.read()

def call_openai(model_name, system_prompt, task_prompt, user_input, temperature=0):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it before running the agent.")
    
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt + "\n\n" + user_input}
    ]
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content


def try_parse_json_with_retry(
    llm_response: str,
    model_name: str,
    system_prompt: str,
    task_prompt: str,
    user_input: str,
    temperature: float
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
    print("📌 Stage 1: Parsing original LLM response...")
    try:
        parsed = json.loads(llm_response)
        print("✅ Success on original response")
        return parsed, "original"
    except json.JSONDecodeError as e:
        print(f"❌ Original response failed: {e}")
    
    # Stage 2: Retry - call LLM again with same prompt
    print("📌 Stage 2: Retrying with same prompt...")
    try:
        retry_response = call_openai(
            model_name=model_name,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            user_input=user_input,
            temperature=temperature
        )
        parsed = json.loads(retry_response)
        print("✅ Success on retry")
        return parsed, "retry"
    except json.JSONDecodeError as e:
        print(f"❌ Retry failed: {e}")
    
    # Stage 3: Repair - ask LLM to fix the broken JSON
    print("📌 Stage 3: Using repair prompt to fix JSON...")
    repair_task = f"""The following response was not valid JSON. Please fix it to be valid JSON matching the expected structure.

Broken response:
{llm_response}

Return ONLY the fixed JSON object, nothing else."""
    
    try:
        repaired_response = call_openai(
            model_name=model_name,
            system_prompt=system_prompt,
            task_prompt=repair_task,
            user_input="",  # Empty since repair task is self-contained
            temperature=temperature
        )
        parsed = json.loads(repaired_response)
        print("✅ Success on repair")
        return parsed, "repair"
    except json.JSONDecodeError as e:
        print(f"❌ Repair failed: {e}")
        raise ValueError(
            f"Failed to get valid JSON after 3 attempts (original, retry, repair). "
            f"Last response: {repaired_response[:200]}"
        )

def run_agent(agent_name: str, user_input: dict):
    try:
        config = load_agent_config(agent_name)

        # Load prompts
        system_prompt = load_prompt(config["prompts"]["system"])
        task_prompt = load_prompt(config["prompts"]["task"])

        # Load input schema
        input_schema_path = config["input_schema"]
        
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
        
        initial_response = call_openai(
            model_name=model_name,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            user_input=input_value,
            temperature=temperature
        )

        # Try to parse LLM output with retry and repair mechanism
        parsed, stage = try_parse_json_with_retry(
            llm_response=initial_response,
            model_name=model_name,
            system_prompt=system_prompt,
            task_prompt=task_prompt,
            user_input=input_value,
            temperature=temperature
        )
        
        print(f"📊 JSON parsing succeeded at stage: {stage}")

        # Load and validate against output schema from config
        output_schema_path = config.get("output_schema")
        output_schema = {}
        
        if output_schema_path and os.path.exists(output_schema_path):
            with open(output_schema_path, "r") as f:
                output_schema = json.load(f)
        
        # Validate output fields if schema exists
        if output_schema and isinstance(output_schema, dict):
            for field, field_info in output_schema.items():
                if field not in parsed:
                    raise ValueError(f"Missing field in output: '{field}'")

                value = parsed[field]
                expected_type = field_info.get("type") if isinstance(field_info, dict) else "string"
                
                if expected_type == "string" and not isinstance(value, str):
                    raise ValueError(f"Field '{field}' must be a string.")
                elif expected_type == "integer" and not isinstance(value, int):
                    raise ValueError(f"Field '{field}' must be an integer.")
                elif expected_type.startswith("list[") and not isinstance(value, list):
                    raise ValueError(f"Field '{field}' must be a list.")
                # Optional: check list element types
                if expected_type == "list[string]":
                    if not all(isinstance(item, str) for item in value):
                        raise ValueError(f"All items in '{field}' must be strings.")

        return parsed
    
    except Exception as e:
        print(f"ERROR in run_agent: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise
