AUTOFIT_INPUT_SCHEMA = {
    "type": "object",
    "required": ["csv_path", "target_column"],
    "properties": {
        "csv_path": {"type": "string"},
        "target_column": {"type": "string"},
        "config": {
            "type": "object",
            "properties": {
                "numeric_strategy": {"type": "string", "enum": ["mean", "median"]},
                "scaler": {"type": ["string", "null"], "enum": ["standard", "minmax", None]},
                "categorical_encoding": {"type": "string", "enum": ["onehot", "label"]},
                "search_strategy": {"type": "string", "enum": ["grid", "random"]},
                "cv_splits": {"type": "integer", "minimum": 2},
                "random_iter": {"type": "integer", "minimum": 1},
                "seed": {"type": "integer"},
            },
        },
    },
}


AUTOFIT_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["status", "input", "dataset_summary", "search_summary", "best_model"],
    "properties": {
        "status": {"type": "string"},
        "input": {"type": "object"},
        "dataset_summary": {"type": "object"},
        "eda_summary": {"type": "object"},
        "preprocessing_summary": {"type": "object"},
        "search_summary": {"type": "object"},
        "best_model": {"type": "object"},
        "explainability": {"type": "object"},
        "warnings": {"type": "array"},
    },
}
