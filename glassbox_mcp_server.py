from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def main() -> None:
    _ensure_repo_root_on_path()
    from ironclaw_integration.mcp_server import main as server_main

    server_main()


if __name__ == "__main__":
    main()
