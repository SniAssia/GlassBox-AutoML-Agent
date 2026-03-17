"""Enhanced run_autofit with better WASM compatibility"""

import json
import pandas as pd
import tempfile
import os
from typing import Optional, Union

from Orchestrator.Orchestrator import GlassBoxAutoML
from Orchestrator.GridSearch import GridSearch
from Orchestrator.RandomSearch import RandomSearch
from Orchestrator.KFoldCV import KFoldCV
from models.KNN import KNearestNeighbors

def convert_numpy(obj):
    """Recursively convert NumPy types to native Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(v) for v in obj]
    elif hasattr(obj, "item"):  # handles np.int64, np.float64, etc.
        return obj.item()
    else:
        return obj

def run_autofit(
    dataset_csv: Union[str, pd.DataFrame], 
    target_column: str, 
    search_strategy: str = "grid", 
    n_splits: int = 5
):
    """
    Run AutoML pipeline
    
    Args:
        dataset_csv: Either path to CSV file or pandas DataFrame
        target_column: Name of target column
        search_strategy: 'grid' or 'random'
        n_splits: Number of cross-validation folds
    """
    try:
        # Handle different input types
        if isinstance(dataset_csv, pd.DataFrame):
            df = dataset_csv
        elif isinstance(dataset_csv, str):
            # Check if file exists (might not in WASM)
            if os.path.exists(dataset_csv):
                df = pd.read_csv(dataset_csv)
            else:
                # Assume it's CSV content as string
                from io import StringIO
                df = pd.read_csv(StringIO(dataset_csv))
        else:
            raise ValueError("dataset_csv must be path, DataFrame, or CSV string")

        # Validate target column
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")

        # Prepare features and target
        X = df.drop(columns=[target_column]).values
        y = df[target_column].values

        # Cross-validation
        cv = KFoldCV(n_splits=n_splits, shuffle=True)

        # Choose search strategy
        if search_strategy.lower() == "grid":
            searcher = GridSearch(
                model_class=KNearestNeighbors,
                param_grid={
                    "k": [3, 5, 7],
                    "distance_metric": ["euclidean", "manhattan"]
                }
            )
        elif search_strategy.lower() == "random":
            searcher = RandomSearch(
                model_class=KNearestNeighbors,
                param_grid={
                    "k": [3, 5, 7],
                    "distance_metric": ["euclidean", "manhattan"]
                },
                n_iter=5
            )
        else:
            raise ValueError("search_strategy must be 'grid' or 'random'")

        # Orchestrator
        automl = GlassBoxAutoML(
            search_strategy=searcher,
            cross_validator=cv
        )

        # Run pipeline
        best_score, best_params = automl.run(X, y)

        # Convert NumPy types
        best_score = float(best_score)
        best_params = convert_numpy(best_params)

        # Build JSON report
        report = {
            "best_model": "KNearestNeighbors",
            "hyperparameters": best_params,
            "best_score": best_score,
            "dataset_shape": df.shape,
            "feature_names": list(df.drop(columns=[target_column]).columns),
            "target_column": target_column,
            "cross_validation_folds": n_splits,
            "search_strategy": search_strategy,
            "explanations": [
                f"Best hyperparameters found: {best_params}",
                f"Cross-validation score: {best_score:.4f}",
                f"Model was trained on {df.shape[0]} samples with {df.shape[1]-1} features"
            ]
        }

        return json.dumps(report)

    except Exception as e:
        return json.dumps({"error": str(e)})