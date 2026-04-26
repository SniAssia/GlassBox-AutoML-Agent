import csv
from pathlib import Path

import numpy as np

from Orchestrator.model_selector import ModelSelector
from cleaner.preprocessing_pipeline import PreprocessingPipeline
from eda.report_builder import build_eda_report
from evaluation.report_formatter import build_final_report
from models.model_registry import get_models_for_task


def load_csv_dataset(csv_path):
    csv_path = Path(csv_path)
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError("CSV file is empty")

    fieldnames = reader.fieldnames or []
    matrix = np.array([[row.get(name) for name in fieldnames] for row in rows], dtype=object)
    return fieldnames, matrix


def _normalize_missing_values(matrix):
    normalized = matrix.copy().astype(object)
    missing_tokens = {"", "na", "n/a", "nan", "null", "none", "missing"}
    for row_idx in range(normalized.shape[0]):
        for col_idx in range(normalized.shape[1]):
            value = normalized[row_idx, col_idx]
            if isinstance(value, str) and value.strip().lower() in missing_tokens:
                normalized[row_idx, col_idx] = None
    return normalized


def _infer_task(y):
    values = np.array([value for value in y if value is not None], dtype=object)
    if len(values) == 0:
        raise ValueError("Target column contains only missing values")

    unique_values = np.unique(values)
    try:
        numeric_values = values.astype(float)
        if len(unique_values) <= min(20, max(2, len(values) // 10)):
            return "classification", values
        return "regression", numeric_values
    except (ValueError, TypeError):
        return "classification", values


def run_autofit(csv_path, target_column, config=None):
    config = config or {}
    csv_path = Path(csv_path)
    feature_names, raw_matrix = load_csv_dataset(csv_path)
    if target_column not in feature_names:
        raise ValueError(f"Unknown target column '{target_column}'")

    matrix = _normalize_missing_values(raw_matrix)
    target_idx = feature_names.index(target_column)
    X = np.delete(matrix, target_idx, axis=1)
    y_raw = matrix[:, target_idx]
    X_names = [name for idx, name in enumerate(feature_names) if idx != target_idx]

    task, y = _infer_task(y_raw)
    eda_report = build_eda_report(X, feature_names=X_names)

    pipeline = PreprocessingPipeline(
        numeric_strategy=config.get("numeric_strategy", "mean"),
        scaler=config.get("scaler", "standard"),
        categorical_encoding=config.get("categorical_encoding", "onehot"),
    )
    X_processed = pipeline.fit_transform(X, feature_names=X_names)
    selector = ModelSelector(
        task=task,
        search_strategy=config.get("search_strategy", "random"),
        # 5 
        cv_splits=config.get("cv_splits"),
        # 10 
        random_iter=config.get("random_iter"),
        seed=config.get("seed", 42),
    )
    selection_report = selector.select(X_processed, y)
    return build_final_report(
        csv_path=csv_path,
        target_column=target_column,
        task=task,
        raw_matrix=matrix,
        raw_feature_names=X_names,
        X_processed=X_processed,
        y=y,
        eda_report=eda_report,
        preprocessing_summary=pipeline.get_metadata(),
        selection_report=selection_report,
    )
