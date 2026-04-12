from __future__ import annotations

import json
import os
import sys
from typing import Any

from .json_utils import to_jsonable
from .tool_api import inspect_csv, run_search


_OUTPUT_MODE: str = "content_length"  # or "newline"


def _log(message: str) -> None:
    path = os.environ.get("GLASSBOX_MCP_LOG")
    if not path:
        return

    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(message.rstrip("\n") + "\n")
    except Exception:
        # never let logging break protocol
        return


def _read_exact(n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sys.stdin.buffer.read(n - len(buf))
        if not chunk:
            raise EOFError
        buf += chunk
    return buf


def _read_message() -> dict[str, Any]:
    """Read one JSON-RPC message from stdin.

    Supports:
      - LSP/MCP style framing with Content-Length headers (CRLF or LF)
      - Newline-delimited JSON as a fallback
    """

    first = sys.stdin.buffer.readline()
    if not first:
        raise EOFError

    _log(f"first_line={first!r}")

    stripped = first.strip()
    if stripped.startswith(b"{") or stripped.startswith(b"["):
        global _OUTPUT_MODE
        _OUTPUT_MODE = "newline"
        msg = json.loads(stripped.decode("utf-8"))
        _log(f"mode=newline_json keys={list(msg.keys())}")
        return msg

    # Header mode
    content_length: int | None = None

    line = first
    while True:
        s = line.decode("utf-8", errors="replace").strip()
        if not s:
            break
        if s.lower().startswith("content-length:"):
            content_length = int(s.split(":", 1)[1].strip())
            _log(f"content_length={content_length}")
        line = sys.stdin.buffer.readline()
        if not line:
            raise EOFError

    if content_length is None:
        raise ValueError("Missing Content-Length header")

    body = _read_exact(content_length)
    _log(f"body_prefix={body[:100]!r}")
    msg = json.loads(body.decode("utf-8"))
    _log(f"mode=content_length keys={list(msg.keys())}")
    return msg


def _send(obj: dict[str, Any]) -> None:
    payload_text = json.dumps(obj, ensure_ascii=False)
    if _OUTPUT_MODE == "newline":
        sys.stdout.write(payload_text + "\n")
        sys.stdout.flush()
        return

    payload = payload_text.encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(payload)}\r\n\r\n".encode("utf-8"))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def _result(id_: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _error(id_: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}


TOOLS = {
    "glassbox.inspect_csv": {
        "description": "Inspect CSV (EDA + typing + correlation) and return options for the LLM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "target_variable": {"type": "string"},
                "max_rows": {"type": ["integer", "null"]},
            },
            "required": ["file_path", "target_variable"],
        },
        "handler": lambda args: inspect_csv(args["file_path"], args["target_variable"], max_rows=args.get("max_rows")),
    },
    "glassbox.run_search": {
        "description": "Run preprocessing + hyperparameter search using an explicit config.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "target_variable": {"type": "string"},
                "config": {"type": "object"},
            },
            "required": ["file_path", "target_variable", "config"],
        },
        "handler": lambda args: run_search(args["file_path"], args["target_variable"], args["config"]),
    },
}


def main() -> None:
    _log("server_started")
    while True:
        try:
            msg = _read_message()
        except EOFError:
            return
        except Exception as e:
            _log(f"read_message_error={type(e).__name__}: {e}")
            # can't reliably respond without an id
            continue

        id_ = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        # Notifications have no id; do not respond.
        if id_ is None:
            if method:
                _log(f"notification_ignored={method}")
            continue

        try:
            if method == "initialize":
                _send(
                    _result(
                        id_,
                        {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "glassbox-mcp", "version": "0.1.0"},
                            "capabilities": {"tools": {}},
                        },
                    )
                )
            elif method == "tools/list":
                tools_list = [
                    {"name": name, "description": spec["description"], "inputSchema": spec["inputSchema"]}
                    for name, spec in TOOLS.items()
                ]
                _send(_result(id_, {"tools": tools_list}))
            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments") or {}
                if tool_name not in TOOLS:
                    _send(_error(id_, -32601, f"Unknown tool: {tool_name}"))
                    continue
                res = TOOLS[tool_name]["handler"](arguments)
                res = to_jsonable(res)
                _send(_result(id_, {"content": [{"type": "text", "text": json.dumps(res, ensure_ascii=False)}]}))
            elif method in ("shutdown", "exit"):
                _send(_result(id_, {}))
                return
            else:
                _send(_error(id_, -32601, f"Unknown method: {method}"))
        except Exception as e:
            _log(f"tool_error={type(e).__name__}: {e}")
            # Put the exception message in the top-level error message because
            # some clients only surface `error.message` and ignore `error.data`.
            _send(_error(id_, -32000, f"Tool error: {e}", {"message": str(e)}))


if __name__ == "__main__":
    main()
