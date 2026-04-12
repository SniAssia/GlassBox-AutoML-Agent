import os
import sys
import json
import pandas as pd
import numpy as np

# Add the project root to the Python path.
# This allows the script to find and import modules from the parent project
# when executed from a different directory (e.g., by the IronClaw agent).
# Using a hardcoded absolute path for reliability.
project_root = r'C:\Users\salim\Documents\Projects\AI\GlassBox-AutoML-Agent'
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cleaner.simple_imputer import SimpleImputer
from cleaner.standard_Scaler import StandardScaler
from cleaner.oneHotEncoder import OneHotEncoder
from cleaner.label_encoder import LabelEncoder
from eda.statistics import mean, median, mode, stdev, skewness, kurtosis
from eda.association import pearson_correlation_matrix
from eda.iqr import cap_outliers_iqr
from eda.auto_typing import infer_type
from models.logistic_regression import LogisticRegression
from Orchestrator.GridSearch import GridSearch
from Orchestrator.KFoldCV import KFoldCV
from Orchestrator.Orchestrator import GlassBoxAutoML

def calculate_statistics(df):
    """Calculates summary statistics for numerical columns in a DataFrame."""
    stats = {}
    numerical_df = df.select_dtypes(include=np.number)
    for col in numerical_df.columns:
        column_data = numerical_df[col].dropna()
        if not column_data.empty:
            stats[col] = {
                'mean': mean(column_data),
                'median': median(column_data),
                'mode': mode(column_data),
                'stdev': stdev(column_data) if len(column_data) > 1 else 0,
                'skewness': skewness(column_data) if len(column_data) > 2 else 0,
                'kurtosis': kurtosis(column_data) if len(column_data) > 3 else 0
            }
    return stats

def convert_numpy_types(obj):
    """Recursively converts numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj

def autofit(file_path: str, target_variable: str) -> str:
    """
    Runs the full AutoML pipeline.
    """
    df = pd.read_csv(file_path)
    
    # 1. EDA
    stats = calculate_statistics(df)
    correlation_matrix = pearson_correlation_matrix(df.select_dtypes(include=np.number).values)
    
    # Auto-typing
    typed_df = df.copy()
    column_types = {col: infer_type(df[col]) for col in df.columns}
    for col, col_type in column_types.items():
        if col_type == 'numerical':
            typed_df[col] = pd.to_numeric(typed_df[col], errors='coerce')
        elif col_type == 'boolean':
            typed_df[col] = typed_df[col].astype(bool)
    
    # Outlier handling (on numerical columns)
    numerical_cols = [col for col, col_type in column_types.items() if col_type == 'numerical']
    for col in numerical_cols:
        typed_df[col] = cap_outliers_iqr(typed_df[col])

    X = typed_df.drop(columns=[target_variable])
    y = typed_df[target_variable]

    # 2. Cleaning
    # Impute missing values
    imputer = SimpleImputer(method='mean')
    X_imputed = imputer.fit_transform(X.values) # Pass numpy array to imputer
    X = pd.DataFrame(X_imputed, columns=X.columns) # Convert back to DataFrame
    
    # Identify categorical and numerical features after imputation
    final_column_types = {col: infer_type(X[col]) for col in X.columns}
    
    categorical_features = [col for col, type in final_column_types.items() if type == 'categorical']
    numerical_features = [col for col, type in final_column_types.items() if type == 'numerical']

    # One-Hot Encode categorical features
    if categorical_features:
        encoder = OneHotEncoder()
        # Use .toarray() if the output is sparse
        X_encoded_data = encoder.fit_transform(X[categorical_features])
        if not isinstance(X_encoded_data, np.ndarray):
             X_encoded_data = X_encoded_data.toarray()
        X_encoded_df = pd.DataFrame(X_encoded_data, columns=encoder.get_feature_names_out(categorical_features), index=X.index)
        X = pd.concat([X.drop(columns=categorical_features), X_encoded_df], axis=1)


    # Scale numerical features
    if numerical_features:
        scaler = StandardScaler()
        X[numerical_features] = scaler.fit_transform(X[numerical_features].values)

    # Encode target variable if it's categorical
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # 3. Modeling & Orchestration
    model = LogisticRegression() 
    param_grid = {
        'lr': [0.01, 0.1],
        'n_epochs': [100, 200]
    }
    
    kfold = KFoldCV(n_splits=3)
    grid_search = GridSearch(model, param_grid)
    orchestrator = GlassBoxAutoML(grid_search, kfold)
    
    best_score, best_params = orchestrator.run(X.values, y)

    # 4. Prepare JSON output
    results = {
        "eda_summary": {
            "statistics": stats,
            "correlation_matrix": pd.DataFrame(correlation_matrix).to_dict()
        },
        "best_model_found": {
            "model_name": model.__class__.__name__,
            "best_score": best_score,
            "best_parameters": best_params
        }
    }
    
    # Convert numpy types for JSON serialization
    results = convert_numpy_types(results)
    
    return json.dumps(results, indent=4)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python autofit.py <file_path> <target_variable>")
        sys.exit(1)
        
    file_path = sys.argv[1]
    target_variable = sys.argv[2]
    
    output = autofit(file_path, target_variable)
    print(output)
