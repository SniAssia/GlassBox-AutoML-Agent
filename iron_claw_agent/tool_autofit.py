import json

from autofit import run_autofit
from iron_claw_agent.schemas import AUTOFIT_INPUT_SCHEMA, AUTOFIT_OUTPUT_SCHEMA


def get_tool_definition():
    return {
        "name": "AutoFit",
        "description": "Run the GlassBox AutoML pipeline on a CSV file and return a structured JSON report.",
        "input_schema": AUTOFIT_INPUT_SCHEMA,
        "output_schema": AUTOFIT_OUTPUT_SCHEMA,
    }


def run_tool(payload):
    if "csv_path" not in payload:
        raise ValueError("Missing required field 'csv_path'")
    if "target_column" not in payload:
        raise ValueError("Missing required field 'target_column'")

    config = payload.get("config", {})
    return run_autofit(payload["csv_path"], payload["target_column"], config=config)


if __name__ == "__main__":
    import sys

    payload = json.load(sys.stdin)
    result = run_tool(payload)
    print(json.dumps(result, indent=2))
