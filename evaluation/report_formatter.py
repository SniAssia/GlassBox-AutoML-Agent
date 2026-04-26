import numpy as np

from evaluation.evaluator import Evaluator
from evaluation.explainability import build_explainability_report
from confusion import confusion_matrix

def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_json_ready(value):
    if isinstance(value, dict):
        return {key: _make_json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_make_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_make_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_make_json_ready(item) for item in value.tolist()]
    return _to_builtin(value)


def _target_preview(y, limit=5):
    preview = []
    for value in y[:limit]:
        preview.append(_to_builtin(value))
    return preview


def _format_tried_models(task, tried_models):
    formatted = []
    for item in tried_models:
        if "error" in item:
            formatted.append(item)
            continue

        if task == "regression":
            formatted.append(
                {
                    "model_name": item["model_name"],
                    "cv_mse": float(-item["cv_score"]),
                    "ranking_score": float(item["cv_score"]),
                    "best_params": item["best_params"],
                }
            )
        else:
            formatted.append(item)
    return formatted


def _format_best_model_summary(task, selection_report, metrics):
    summary = {
        "name": selection_report["best_model_name"],
        "best_params": selection_report["best_params"],
        "train_metrics": metrics,
    }

    if task == "regression":
        summary["cv_mse"] = float(-selection_report["best_score"])
        summary["ranking_score"] = float(selection_report["best_score"])
    else:
        summary["cv_score"] = float(selection_report["best_score"])

    return summary


# ... (existing imports)
from confusion import confusion_matrix # Import your new function

# ... (keep helper functions like _to_builtin, _make_json_ready, etc. as they are)

def build_final_report(
    csv_path,
    target_column,
    task,
    raw_matrix,
    raw_feature_names,
    X_processed,
    y,
    eda_report,
    preprocessing_summary,
    selection_report,
):
    evaluator = Evaluator()
    best_model = selection_report["best_model"]
    y_pred = best_model.predict(X_processed)

    # --- UPDATED SECTION ---
    conf_matrix_data = None
    if task == "classification":
        metrics = evaluator.classification_report(np.asarray(y), np.asarray(y_pred))
        
        # Call your custom confusion matrix function
        matrix, classes = confusion_matrix(np.asarray(y), np.asarray(y_pred))
        conf_matrix_data = {
            "matrix": matrix,
            "labels": classes
        }
    else:
        metrics = evaluator.regression_report(np.asarray(y, dtype=float), np.asarray(y_pred, dtype=float))
    # -----------------------

    explainability = build_explainability_report(
        selection_report["best_model_name"],
        best_model,
        preprocessing_summary["output_features"],
    )

    report = {
        "status": "success",
        "input": {
            "csv_path": str(csv_path),
            "target_column": target_column,
            "task": task,
        },
        "dataset_summary": {
            "rows": int(raw_matrix.shape[0]),
            "columns": int(raw_matrix.shape[1]),
            "raw_feature_names": raw_feature_names,
            "processed_feature_names": preprocessing_summary["output_features"],
            "processed_feature_count": int(X_processed.shape[1]),
            "target_preview": _target_preview(y),
        },
        "eda_summary": eda_report,
        "preprocessing_summary": preprocessing_summary,
        "search_summary": {
            "strategy": selection_report["search_strategy"],
            "cv_splits": selection_report["cv_splits"],
            "tried_models": _format_tried_models(task, selection_report["tried_models"]),
        },
        "best_model": _format_best_model_summary(task, selection_report, metrics),
        "explainability": explainability,
        "confusion_matrix": conf_matrix_data, # Added to the report
        "warnings": [],
    }
    return _make_json_ready(report)