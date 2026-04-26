import importlib.util
from pathlib import Path

import numpy as np

from eda.association import pearson_correlation_matrix
from eda.iqr import iqr_bounds, iqr_outliers
from eda.statistics import kurtosis, max_val, mean, median, min_val, mode, skewness, stdev


def _load_auto_typing_module():
    module_path = Path(__file__).with_name("auto-typing.py")
    spec = importlib.util.spec_from_file_location("eda_auto_typing", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_AUTO_TYPING = _load_auto_typing_module()


def _is_missing(value):
    return value is None or (isinstance(value, float) and np.isnan(value))


def _non_missing_values(column):
    return np.array([value for value in column if not _is_missing(value)], dtype=object)


def _to_numeric(values):
    if len(values) == 0:
        return np.array([], dtype=float)
    return np.array(values, dtype=float)


def _safe_stat(stat_fn, values, minimum_size):
    if len(values) < minimum_size:
        return None
    return float(stat_fn(values))


def infer_column_types(X, feature_names=None):
    X = np.asarray(X, dtype=object)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    names = feature_names or [f"feature_{idx}" for idx in range(X.shape[1])]
    inferred = {}
    for idx, name in enumerate(names):
        values = _non_missing_values(X[:, idx])
        inferred[name] = "categorical" if len(values) == 0 else _AUTO_TYPING.infer_type(values)
    return inferred


def build_eda_report(X, feature_names=None):
    X = np.asarray(X, dtype=object)
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")

    names = feature_names or [f"feature_{idx}" for idx in range(X.shape[1])]
    column_types = infer_column_types(X, names)
    column_reports = []
    numeric_columns = []
    numeric_names = []

    for idx, name in enumerate(names):
        column = X[:, idx]
        values = _non_missing_values(column)
        col_type = column_types[name]
        report = {
            "name": name,
            "inferred_type": col_type,
            "missing_count": int(sum(_is_missing(value) for value in column)),
            "non_missing_count": int(len(values)),
        }

        if col_type == "numerical" and len(values) > 0:
            numeric_values = _to_numeric(values)
            lower, upper = iqr_bounds(numeric_values)
            report["summary"] = {
                "mean": float(mean(numeric_values)),
                "median": float(median(numeric_values)),
                "std": _safe_stat(stdev, numeric_values, minimum_size=2),
                "min": float(min_val(numeric_values)),
                "max": float(max_val(numeric_values)),
                "skewness": _safe_stat(skewness, numeric_values, minimum_size=3),
                "kurtosis": _safe_stat(kurtosis, numeric_values, minimum_size=4),
            }
            report["outliers"] = {
                "count": int(len(iqr_outliers(numeric_values))),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
            }
            numeric_columns.append(np.array([np.nan if _is_missing(v) else float(v) for v in column], dtype=float))
            numeric_names.append(name)
        elif len(values) > 0:
            report["summary"] = {
                "unique_count": int(len(np.unique(values))),
                "mode": mode(values),
            }
        else:
            report["summary"] = {"note": "column contains only missing values"}

        column_reports.append(report)

    correlation_summary = {"numeric_features": numeric_names, "matrix": []}
    if len(numeric_columns) >= 2:
        numeric_matrix = np.column_stack(numeric_columns)
        valid_rows = ~np.isnan(numeric_matrix).any(axis=1)
        if np.any(valid_rows):
            corr = pearson_correlation_matrix(numeric_matrix[valid_rows])
            correlation_summary["matrix"] = np.asarray(corr, dtype=float).tolist()

    return {
        "row_count": int(X.shape[0]),
        "column_count": int(X.shape[1]),
        "feature_names": names,
        "column_types": column_types,
        "columns": column_reports,
        "correlation_summary": correlation_summary,
    }
