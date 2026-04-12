from __future__ import annotations

import math
import random
from dataclasses import dataclass
from itertools import product
from typing import Any, Optional

import numpy as np

from cleaner.label_encoder import LabelEncoder
from cleaner.minMax_Scaler import MinMaxScaler
from cleaner.oneHotEncoder import OneHotEncoder
from cleaner.simple_imputer import SimpleImputer
from cleaner.standard_Scaler import StandardScaler
from eda.auto_typing import infer_type
from eda.iqr import iqr_bounds
from eda.statistics import kurtosis, mean, median, mode, skewness, stdev
from evaluation.confusion import confusion_matrix
from evaluation.metrics_classification import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    weighted_f1_score,
)
from evaluation.metrics_regression import mean_absolute_error, mean_squared_error, r2_score
from models.KNN import KNearestNeighbors
from models.decision_tree import DecisionTree
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.naive_bayes import GaussianNaiveBayes
from models.random_forest import RandomForest

from .csv_utils import LoadedCSV, iter_columns, load_csv


def _set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    np.random.seed(seed)
    random.seed(seed)


def _non_missing(values: list[object]) -> list[object]:
    return [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]


def _to_float_list(values: list[object]) -> list[float]:
    out: list[float] = []
    for v in values:
        if v is None:
            continue
        if isinstance(v, (int, float, np.integer, np.floating)):
            if isinstance(v, float) and math.isnan(v):
                continue
            out.append(float(v))
        else:
            try:
                out.append(float(str(v)))
            except ValueError:
                continue
    return out


def _pairwise_pearson_ignore_missing(x: list[object], y: list[object]) -> Optional[float]:
    if len(x) != len(y):
        raise ValueError("x and y must have same length")

    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(x, y):
        if a is None or b is None:
            continue
        try:
            fa = float(a)
            fb = float(b)
        except Exception:
            continue
        if math.isnan(fa) or math.isnan(fb):
            continue
        xs.append(fa)
        ys.append(fb)

    if len(xs) <= 1:
        return None

    xa = np.asarray(xs, dtype=float)
    ya = np.asarray(ys, dtype=float)

    mu_x = mean(xa)
    mu_y = mean(ya)
    sx = stdev(xa, mu_x) if xa.size > 1 else 0.0
    sy = stdev(ya, mu_y) if ya.size > 1 else 0.0
    if sx == 0 or sy == 0:
        return 0.0

    covariance = np.sum((xa - mu_x) * (ya - mu_y)) / (xa.size - 1)
    return float(covariance / (sx * sy))


def _numeric_stats(values: list[object]) -> Optional[dict[str, Any]]:
    xs = _to_float_list(values)
    if len(xs) == 0:
        return None

    arr = np.asarray(xs, dtype=float)
    result: dict[str, Any] = {
        "count": int(arr.size),
        "mean": mean(arr),
        "median": median(arr),
        "mode": mode(arr),
    }

    if arr.size > 1:
        result["stdev"] = stdev(arr)
    if arr.size > 2:
        result["skewness"] = skewness(arr)
    if arr.size > 3:
        result["kurtosis"] = kurtosis(arr)

    return result


def _iqr_outlier_summary(values: list[object], *, factor: float = 1.5) -> Optional[dict[str, Any]]:
    xs = _to_float_list(values)
    if len(xs) == 0:
        return None

    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return None

    lower, upper = iqr_bounds(arr, factor=factor)
    mask = (arr < lower) | (arr > upper)

    return {
        "factor": factor,
        "lower": float(lower),
        "upper": float(upper),
        "outlier_count": int(np.sum(mask)),
        "outlier_fraction": float(np.mean(mask)),
    }


def _infer_task(target_values: list[object]) -> str:
    # heuristic: numeric -> regression, else classification
    t = infer_type(np.asarray(_non_missing(target_values), dtype=object))
    return "regression" if t == "numerical" else "classification"


def _default_options(task: str) -> dict[str, Any]:
    # Keep this small and auditable: a short list of supported choices.
    models: dict[str, Any] = {
        "classification": {
            "logistic_regression": {"params": {"lr": [0.01, 0.1], "n_epochs": [100, 200]}},
            "knn": {"params": {"k": [3, 5, 7], "distance_metric": ["euclidean", "manhattan"]}},
            "decision_tree": {"params": {"max_depth": [3, 5, 10], "min_samples_split": [2, 5]}},
            "random_forest": {"params": {"n_trees": [10, 25], "max_depth": [5, 10]}},
            "naive_bayes": {"params": {"var_smoothing": [1e-9, 1e-6]}},
        },
        "regression": {
            "linear_regression": {"params": {"solver": ["gd", "normal"], "lr": [0.01, 0.1], "n_epochs": [200, 500]}},
            "knn": {"params": {"k": [3, 5, 7], "distance_metric": ["euclidean", "manhattan"]}},
            "decision_tree": {"params": {"max_depth": [3, 5, 10], "min_samples_split": [2, 5]}},
            "random_forest": {"params": {"n_trees": [10, 25], "max_depth": [5, 10]}},
        },
    }

    return {
        "preprocess": {
            "imputer": {"method": ["mean", "median"]},
            "outliers": {"strategy": ["none", "cap"], "factor": [1.5]},
            "encoding": {"categorical": ["onehot", "label", "none"]},
            "scaling": {"numerical": ["none", "standard", "minmax"]},
        },
        "search": {
            "strategy": ["grid", "random"],
            "cv": {"n_splits": [3, 5], "shuffle": [True], "seed": [42]},
            "random": {"n_iter": [10, 25]},
        },
        "models": models[task],
    }


def inspect_csv(file_path: str, target_variable: str, *, max_rows: Optional[int] = None) -> dict[str, Any]:
    loaded = load_csv(file_path, max_rows=max_rows)

    if target_variable not in loaded.header:
        raise ValueError(f"target_variable '{target_variable}' not found in CSV header")

    target_idx = loaded.header.index(target_variable)

    column_profiles: dict[str, Any] = {}
    types: dict[str, str] = {}

    for name, values in iter_columns(loaded):
        inferred = infer_type(np.asarray(_non_missing(values), dtype=object)) if _non_missing(values) else "categorical"
        types[name] = inferred
        non_missing_vals = _non_missing(values)
        unique_non_missing = len(set(map(str, non_missing_vals)))

        profile: dict[str, Any] = {
            "inferred_type": inferred,
            "missing_count": int(sum(v is None for v in values)),
            "unique_non_missing": int(unique_non_missing),
        }

        if inferred == "numerical":
            profile["statistics"] = _numeric_stats(values)
            profile["outliers_iqr"] = _iqr_outlier_summary(values)

        column_profiles[name] = profile

    task = _infer_task([r[target_idx] for r in loaded.rows])

    # Numeric correlation matrix (pairwise, missing-safe)
    numeric_cols = [c for c in loaded.header if types.get(c) == "numerical" and c != target_variable]
    corr_matrix: list[list[Optional[float]]] = []
    for a in numeric_cols:
        row: list[Optional[float]] = []
        av = [r[loaded.header.index(a)] for r in loaded.rows]
        for b in numeric_cols:
            bv = [r[loaded.header.index(b)] for r in loaded.rows]
            row.append(_pairwise_pearson_ignore_missing(av, bv))
        corr_matrix.append(row)

    return {
        "dataset": {
            "file_path": file_path,
            "file_path_resolved": getattr(loaded, "source_path", file_path),
            "n_rows": int(len(loaded.rows)),
            "n_columns": int(len(loaded.header)),
            "columns": list(loaded.header),
            "target_variable": target_variable,
            "task_inferred": task,
        },
        "columns": column_profiles,
        "associations": {
            "numeric_correlation": {
                "columns": numeric_cols,
                "matrix": corr_matrix,
            }
        },
        "options": _default_options(task),
    }


@dataclass
class _EncodedFeatures:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    target_encoder: Optional[LabelEncoder]
    encoders: dict[str, Any]
    task: str


def _cap_outliers_inplace(col: list[object], *, factor: float) -> list[object]:
    xs = _to_float_list(col)
    if len(xs) == 0:
        return col

    lower, upper = iqr_bounds(np.asarray(xs, dtype=float), factor=factor)
    out: list[object] = []
    for v in col:
        if v is None:
            out.append(None)
            continue
        try:
            fv = float(v)
        except Exception:
            out.append(v)
            continue
        if math.isnan(fv):
            out.append(None)
        else:
            out.append(float(np.clip(fv, lower, upper)))
    return out


def _build_design_matrix(loaded: LoadedCSV, target_variable: str, config: dict[str, Any]) -> _EncodedFeatures:
    header = loaded.header
    target_idx = header.index(target_variable)

    preprocess = config.get("preprocess", {})
    outliers_cfg = preprocess.get("outliers", {})
    outlier_strategy = outliers_cfg.get("strategy", "cap")
    outlier_factor = float(outliers_cfg.get("factor", 1.5))

    # Separate raw columns
    columns: dict[str, list[object]] = {name: [r[i] for r in loaded.rows] for i, name in enumerate(header)}

    # Infer types on raw columns (ignoring missing)
    inferred_types: dict[str, str] = {}
    for name, values in columns.items():
        if name == target_variable:
            continue
        non_missing = _non_missing(values)
        inferred_types[name] = infer_type(np.asarray(non_missing, dtype=object)) if non_missing else "categorical"

    # Outlier handling (numeric only)
    if outlier_strategy == "cap":
        for name, t in inferred_types.items():
            if t == "numerical":
                columns[name] = _cap_outliers_inplace(columns[name], factor=outlier_factor)

    # Build X_raw object matrix (excluding target)
    feature_names_in = [c for c in header if c != target_variable]
    X_obj = np.asarray([[columns[c][i] for c in feature_names_in] for i in range(len(loaded.rows))], dtype=object)

    # Impute missing
    imputer_cfg = preprocess.get("imputer", {})
    imputer_method = imputer_cfg.get("method", "mean")
    imputer = SimpleImputer(method=imputer_method)
    X_imp = imputer.fit_transform(X_obj)

    # Encode categorical
    encoding_cfg = preprocess.get("encoding", {})
    cat_encoding = encoding_cfg.get("categorical", "onehot")

    cat_cols = [c for c in feature_names_in if inferred_types.get(c) == "categorical"]
    num_cols = [c for c in feature_names_in if inferred_types.get(c) == "numerical"]
    bool_cols = [c for c in feature_names_in if inferred_types.get(c) == "boolean"]

    encoders: dict[str, Any] = {"imputer": {"method": imputer_method, "fill_values": getattr(imputer, "fill_values_", None)}}

    # start with numerical + boolean (booleans become 0/1)
    X_parts: list[np.ndarray] = []
    feature_names_out: list[str] = []

    if num_cols:
        idxs = [feature_names_in.index(c) for c in num_cols]
        X_num = X_imp[:, idxs].astype(float)
        X_parts.append(X_num)
        feature_names_out.extend(num_cols)

    if bool_cols:
        idxs = [feature_names_in.index(c) for c in bool_cols]
        raw = X_imp[:, idxs]
        X_bool = np.vectorize(lambda v: 1.0 if str(v).strip().lower() in {"1", "true", "yes"} else 0.0)(raw).astype(float)
        X_parts.append(X_bool)
        feature_names_out.extend(bool_cols)

    if cat_cols and cat_encoding != "none":
        if cat_encoding == "onehot":
            ohe_by_col: dict[str, Any] = {}
            for c in cat_cols:
                idx = feature_names_in.index(c)
                col_vals = X_imp[:, idx]
                enc = OneHotEncoder()
                mat = enc.fit_transform(col_vals)
                ohe_by_col[c] = {"classes": enc.classes_.tolist() if enc.classes_ is not None else None}
                names = [f"{c}__{cls}" for cls in (enc.classes_.tolist() if enc.classes_ is not None else [])] + [f"{c}__unknown"]
                X_parts.append(mat.astype(float))
                feature_names_out.extend(names)
            encoders["onehot"] = ohe_by_col
        elif cat_encoding == "label":
            le_by_col: dict[str, Any] = {}
            mats: list[np.ndarray] = []
            names: list[str] = []
            for c in cat_cols:
                idx = feature_names_in.index(c)
                col_vals = X_imp[:, idx]
                enc = LabelEncoder()
                vec = enc.fit_transform(col_vals)
                le_by_col[c] = {"classes": enc.classes_.tolist() if enc.classes_ is not None else None}
                mats.append(vec.reshape(-1, 1).astype(float))
                names.append(c)
            X_parts.append(np.hstack(mats) if mats else np.zeros((X_imp.shape[0], 0)))
            feature_names_out.extend(names)
            encoders["label"] = le_by_col
        else:
            raise ValueError(f"Unsupported categorical encoding: {cat_encoding}")

    X = np.hstack(X_parts) if X_parts else np.zeros((X_imp.shape[0], 0), dtype=float)

    # Scaling (only original numerical columns, i.e., the first len(num_cols) columns)
    scaling_cfg = preprocess.get("scaling", {})
    scaling = scaling_cfg.get("numerical", "standard")
    if num_cols and scaling != "none":
        n_num = len(num_cols)
        if scaling == "standard":
            scaler = StandardScaler()
            X[:, :n_num] = scaler.fit_transform(X[:, :n_num])
            encoders["scaler"] = {"type": "standard", "mean": scaler.mean_.tolist(), "stdev": scaler.stadev_.tolist()}
        elif scaling == "minmax":
            scaler = MinMaxScaler()
            X[:, :n_num] = scaler.fit_transform(X[:, :n_num])
            encoders["scaler"] = {"type": "minmax", "min": scaler.min_.tolist(), "max": scaler.max_.tolist()}
        else:
            raise ValueError(f"Unsupported scaling: {scaling}")

    # Target handling
    y_raw = [r[target_idx] for r in loaded.rows]
    task = config.get("task") or "auto"
    if task == "auto":
        task = _infer_task(y_raw)

    target_encoder: Optional[LabelEncoder] = None
    if task == "classification":
        # encode labels if non-numeric
        non_missing = _non_missing(y_raw)
        y_type = infer_type(np.asarray(non_missing, dtype=object)) if non_missing else "categorical"
        if y_type == "numerical":
            y = np.asarray([int(float(v)) if v is not None else 0 for v in y_raw], dtype=int)
        else:
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(np.asarray([v if v is not None else "" for v in y_raw], dtype=object))
    else:
        y = np.asarray([float(v) if v is not None else np.nan for v in y_raw], dtype=float)
        if np.isnan(y).any():
            raise ValueError("Regression target contains missing values; please impute/drop before training")

    return _EncodedFeatures(
        X=X.astype(float),
        y=y,
        feature_names=feature_names_out,
        target_encoder=target_encoder,
        encoders=encoders,
        task=task,
    )


def _instantiate_model(name: str, task: str, params: dict[str, Any]) -> Any:
    if name == "logistic_regression":
        if task != "classification":
            raise ValueError("logistic_regression only supports classification")
        return LogisticRegression(**params)
    if name == "linear_regression":
        if task != "regression":
            raise ValueError("linear_regression only supports regression")
        return LinearRegression(**params)
    if name == "knn":
        return KNearestNeighbors(task=task, **params)
    if name == "decision_tree":
        return DecisionTree(task=task, **params)
    if name == "random_forest":
        return RandomForest(task=task, **params)
    if name == "naive_bayes":
        if task != "classification":
            raise ValueError("naive_bayes only supports classification")
        return GaussianNaiveBayes(**params)

    raise ValueError(f"Unknown model name: {name}")


def _grid_param_sets(space: dict[str, list[Any]], *, max_combinations: Optional[int] = None) -> list[dict[str, Any]]:
    keys = list(space.keys())
    if not keys:
        return [{}]

    values = [list(space[k]) for k in keys]
    combos: list[dict[str, Any]] = []
    for tup in product(*values):
        combos.append(dict(zip(keys, tup)))
        if max_combinations is not None and len(combos) >= max_combinations:
            break
    return combos


def _random_param_sets(space: dict[str, list[Any]], *, n_iter: int, seed: Optional[int]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    keys = list(space.keys())
    out: list[dict[str, Any]] = []
    for _ in range(n_iter):
        params: dict[str, Any] = {}
        for k in keys:
            choices = list(space[k])
            params[k] = choices[int(rng.integers(0, len(choices)))]
        out.append(params)
    return out


def _evaluate_model_cv(model_name: str, task: str, X: np.ndarray, y: np.ndarray, *,
                       param_sets: list[dict[str, Any]], cv_splits: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    best_score = -np.inf
    best_params: dict[str, Any] = {}
    best_cv_scores: list[float] = []

    for params in param_sets:
        fold_scores: list[float] = []
        for train_idx, val_idx in cv_splits:
            model = _instantiate_model(model_name, task, params)
            model.fit(X[train_idx], y[train_idx])
            fold_scores.append(float(model.score(X[val_idx], y[val_idx])))
        mean_score = float(np.mean(fold_scores))
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
            best_cv_scores = fold_scores

    return {
        "model_name": model_name,
        "best_score": best_score,
        "best_params": best_params,
        "cv_scores": best_cv_scores,
    }


def run_search(file_path: str, target_variable: str, config: dict[str, Any]) -> dict[str, Any]:
    seed = config.get("seed", 42)
    _set_seed(seed)

    loaded = load_csv(file_path)
    if target_variable not in loaded.header:
        raise ValueError(f"target_variable '{target_variable}' not found in CSV header")

    encoded = _build_design_matrix(loaded, target_variable, config)

    search_cfg = config.get("search", {})
    strategy = search_cfg.get("strategy", "grid")

    cv_cfg = search_cfg.get("cv", {})
    n_splits = int(cv_cfg.get("n_splits", 3))
    shuffle = bool(cv_cfg.get("shuffle", True))
    cv_seed = int(cv_cfg.get("seed", seed))

    # build CV splits here to ensure consistent evaluation across candidates
    indices = np.arange(encoded.X.shape[0])
    if shuffle:
        rng = np.random.default_rng(cv_seed)
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, encoded.X.shape[0] // n_splits, dtype=int)
    fold_sizes[: encoded.X.shape[0] % n_splits] += 1

    cv_splits: list[tuple[np.ndarray, np.ndarray]] = []
    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current: current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        cv_splits.append((train_idx, val_idx))
        current += fold_size

    candidates = search_cfg.get("models")
    if not candidates:
        # sensible default if caller didn't provide candidates
        opts = _default_options(encoded.task)
        candidates = [{"name": k, "space": v.get("params", {})} for k, v in opts["models"].items()]

    max_combinations = search_cfg.get("grid", {}).get("max_combinations")
    if max_combinations is not None:
        max_combinations = int(max_combinations)

    n_iter = int(search_cfg.get("random", {}).get("n_iter", 10))

    evaluated: list[dict[str, Any]] = []
    best_overall: Optional[dict[str, Any]] = None

    for cand in candidates:
        model_name = cand.get("name")
        space = cand.get("space") or cand.get("param_grid") or {}

        if strategy == "grid":
            param_sets = _grid_param_sets(space, max_combinations=max_combinations)
        elif strategy == "random":
            param_sets = _random_param_sets(space, n_iter=n_iter, seed=seed)
        else:
            raise ValueError(f"Unsupported search strategy: {strategy}")

        result = _evaluate_model_cv(model_name, encoded.task, encoded.X, encoded.y, param_sets=param_sets, cv_splits=cv_splits)
        evaluated.append(result)

        if best_overall is None or result["best_score"] > best_overall["best_score"]:
            best_overall = result

    assert best_overall is not None

    # Fit best model on full data for a simple final report
    best_model = _instantiate_model(best_overall["model_name"], encoded.task, best_overall["best_params"])
    best_model.fit(encoded.X, encoded.y)
    y_pred = best_model.predict(encoded.X)

    if encoded.task == "classification":
        cm, classes = confusion_matrix(encoded.y, y_pred)
        metrics = {
            "accuracy": float(accuracy_score(encoded.y, y_pred)),
            "precision": float(precision_score(encoded.y, y_pred)),
            "recall": float(recall_score(encoded.y, y_pred)),
            "f1": float(f1_score(encoded.y, y_pred)),
            "weighted_f1": float(weighted_f1_score(encoded.y, y_pred)),
            "confusion_matrix": cm.tolist(),
            "classes": classes.tolist(),
        }
    else:
        metrics = {
            "mae": float(mean_absolute_error(encoded.y, y_pred)),
            "mse": float(mean_squared_error(encoded.y, y_pred)),
            "r2": float(r2_score(encoded.y, y_pred)),
        }

    return {
        "dataset": {
            "file_path": file_path,
            "file_path_resolved": getattr(loaded, "source_path", file_path),
            "target_variable": target_variable,
            "task": encoded.task,
            "n_rows": int(encoded.X.shape[0]),
            "n_features": int(encoded.X.shape[1]),
        },
        "preprocessing": {
            "feature_names_out": encoded.feature_names,
            "encoders": encoded.encoders,
            "target_classes": encoded.target_encoder.classes_.tolist() if encoded.target_encoder is not None else None,
        },
        "search": {
            "strategy": strategy,
            "candidates_evaluated": evaluated,
            "best": best_overall,
        },
        "final_model": {
            "model_name": best_overall["model_name"],
            "params": best_overall["best_params"],
            "train_metrics": metrics,
        },
    }
