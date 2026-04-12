import numpy as np


_MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none"}


def _is_missing(v) -> bool:
    if v is None:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    if isinstance(v, str) and v.strip().lower() in _MISSING_STRINGS:
        return True
    return False


def _clean_non_missing(x):
    x = np.asarray(x, dtype=object)
    return np.array([v for v in x if not _is_missing(v)], dtype=object)


def _normalize_bool_token(v):
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "yes", "y", "t"}:
            return 1
        if s in {"false", "no", "n", "f"}:
            return 0
        if s in {"0", "1"}:
            return int(s)
    return v


def is_boolean(x):
    x = _clean_non_missing(x)
    if x.size == 0:
        return False

    allowed = {0, 1, True, False}
    normalized = [_normalize_bool_token(v) for v in x]
    return all(v in allowed for v in normalized)


def is_numerical(x):
    x = _clean_non_missing(x)
    if x.size == 0:
        return False

    if is_boolean(x):
        return False

    try:
        np.asarray(x, dtype=float)
        return True
    except (ValueError, TypeError):
        return False


def is_categorical(x):
    x = _clean_non_missing(x)
    if x.size == 0:
        return True

    return not is_boolean(x) and not is_numerical(x)


def infer_type(x):
    if is_boolean(x):
        return "boolean"
    if is_numerical(x):
        return "numerical"
    return "categorical"


def infer_types(X):
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("infer_types expects a 2D array")

    return [infer_type(X[:, i]) for i in range(X.shape[1])]