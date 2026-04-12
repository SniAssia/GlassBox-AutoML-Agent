import json
import sys
from pathlib import Path


def _ensure_project_root_on_path() -> None:
    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _load_config(third_arg: str | None) -> dict | None:
    if third_arg is None:
        return None

    s = third_arg.strip()
    if not s:
        return None

    # If it's a path, load it.
    p = Path(s)
    if p.exists() and p.is_file():
        return json.loads(p.read_text(encoding="utf-8-sig"))

    # Otherwise assume JSON.
    return json.loads(s)


def main() -> None:
    _ensure_project_root_on_path()

    from ironclaw_integration.json_utils import dumps
    from ironclaw_integration.tool_api import inspect_csv, run_search

    if len(sys.argv) < 3:
        print("Usage: python autofit.py <file_path> <target_variable> [config_json_or_path]")
        sys.exit(1)

    file_path = sys.argv[1]
    target_variable = sys.argv[2]
    config = _load_config(sys.argv[3] if len(sys.argv) >= 4 else None)

    if config is None:
        report = inspect_csv(file_path, target_variable)
        report["next_step"] = "Re-run with a 3rd argument: JSON config (or a config file path) to execute run_search."
        print(dumps(report, indent=2))
        return

    result = run_search(file_path, target_variable, config)
    print(dumps(result, indent=2))


if __name__ == "__main__":
    main()
