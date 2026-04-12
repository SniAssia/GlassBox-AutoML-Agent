from __future__ import annotations

import argparse
import json
from typing import Any, Optional

from .json_utils import dumps
from .tool_api import inspect_csv, run_search


def _parse_config(config_json: Optional[str], config_path: Optional[str]) -> Optional[dict[str, Any]]:
    if config_path:
        with open(config_path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    if config_json is None:
        return None

    config_json = config_json.strip()
    if not config_json:
        return None

    return json.loads(config_json)


def main() -> None:
    parser = argparse.ArgumentParser(description="GlassBox AutoML tool entrypoint")
    sub = parser.add_subparsers(dest="command", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect a CSV and return an EDA/options JSON")
    p_inspect.add_argument("--file_path", required=True)
    p_inspect.add_argument("--target_variable", required=True)
    p_inspect.add_argument("--max_rows", type=int, default=None)

    p_run = sub.add_parser("run", help="Run search/training using a provided JSON config")
    p_run.add_argument("--file_path", required=True)
    p_run.add_argument("--target_variable", required=True)
    p_run.add_argument("--config_json", default=None)
    p_run.add_argument("--config_path", default=None)

    args = parser.parse_args()

    if args.command == "inspect":
        report = inspect_csv(args.file_path, args.target_variable, max_rows=args.max_rows)
        print(dumps(report, indent=2))
        return

    config = _parse_config(args.config_json, args.config_path)
    if config is None:
        # two-step workflow: without config, just return inspection + a note
        report = inspect_csv(args.file_path, args.target_variable)
        report["next_step"] = "Call run with config_json or config_path to execute search."
        print(dumps(report, indent=2))
        return

    result = run_search(args.file_path, args.target_variable, config)
    print(dumps(result, indent=2))


if __name__ == "__main__":
    main()
