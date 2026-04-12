from __future__ import annotations

import json

from typing import Any

import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Convert NumPy/Python objects into JSON-serializable primitives."""

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def dumps(obj: Any, *, indent: int = 2) -> str:
    return json.dumps(to_jsonable(obj), indent=indent, ensure_ascii=False)
