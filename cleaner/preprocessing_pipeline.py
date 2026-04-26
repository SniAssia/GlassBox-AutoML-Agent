import numpy as np

from cleaner.label_encoder import LabelEncoder
from cleaner.minMax_Scaler import MinMaxScaler
from cleaner.oneHotEncoder import OneHotEncoder
from cleaner.simple_imputer import SimpleImputer
from cleaner.standard_Scaler import StandardScaler
from eda.report_builder import infer_column_types


class PreprocessingPipeline:
    def __init__(self, numeric_strategy="mean", scaler="standard", categorical_encoding="onehot"):
        if numeric_strategy not in ("mean", "median"):
            raise ValueError("numeric_strategy must be 'mean' or 'median'")
        if scaler not in ("standard", "minmax", None):
            raise ValueError("scaler must be 'standard', 'minmax', or None")
        if categorical_encoding not in ("onehot", "label"):
            raise ValueError("categorical_encoding must be 'onehot' or 'label'")

        self.numeric_strategy = numeric_strategy
        self.scaler = scaler
        self.categorical_encoding = categorical_encoding
        self.feature_names_ = None
        self.column_types_ = None
        self.imputer_ = None
        self.numeric_indices_ = []
        self.categorical_indices_ = []
        self.boolean_indices_ = []
        self.numeric_scaler_ = None
        self.encoders_ = {}
        self.output_features_ = []
        self.metadata_ = None

    def fit(self, X, feature_names=None):
        X = np.asarray(X, dtype=object)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.feature_names_ = feature_names or [f"feature_{idx}" for idx in range(X.shape[1])]
        self.column_types_ = infer_column_types(X, self.feature_names_)
        self.imputer_ = SimpleImputer(method=self.numeric_strategy).fit(X)
        X_imputed = self.imputer_.transform(X)

        self.numeric_indices_ = [idx for idx, name in enumerate(self.feature_names_) if self.column_types_[name] == "numerical"]
        self.categorical_indices_ = [idx for idx, name in enumerate(self.feature_names_) if self.column_types_[name] == "categorical"]
        self.boolean_indices_ = [idx for idx, name in enumerate(self.feature_names_) if self.column_types_[name] == "boolean"]

        if self.numeric_indices_ and self.scaler is not None:
            scaler_cls = StandardScaler if self.scaler == "standard" else MinMaxScaler
            numeric_values = X_imputed[:, self.numeric_indices_].astype(float)
            self.numeric_scaler_ = scaler_cls().fit(numeric_values)

        self.encoders_ = {}
        self.output_features_ = []

        for idx in self.numeric_indices_:
            self.output_features_.append(self.feature_names_[idx])

        for idx in self.boolean_indices_:
            self.output_features_.append(self.feature_names_[idx])

        for idx in self.categorical_indices_:
            values = X_imputed[:, idx]
            if self.categorical_encoding == "onehot":
                encoder = OneHotEncoder().fit(values)
                encoded_names = [f"{self.feature_names_[idx]}__{value}" for value in encoder.classes_]
                encoded_names.append(f"{self.feature_names_[idx]}__unknown")
            else:
                encoder = LabelEncoder().fit(values)
                encoded_names = [self.feature_names_[idx]]

            self.encoders_[idx] = encoder
            self.output_features_.extend(encoded_names)

        self.metadata_ = {
            "numeric_strategy": self.numeric_strategy,
            "scaler": self.scaler,
            "categorical_encoding": self.categorical_encoding,
            "column_types": self.column_types_,
            "input_features": self.feature_names_,
            "output_features": self.output_features_,
        }
        return self

    def transform(self, X):
        if self.imputer_ is None:
            raise RuntimeError("PreprocessingPipeline is not fitted yet. Call fit() first.")

        X = np.asarray(X, dtype=object)
        X_imputed = self.imputer_.transform(X)
        transformed_blocks = []

        if self.numeric_indices_:
            numeric_values = X_imputed[:, self.numeric_indices_].astype(float)
            if self.numeric_scaler_ is not None:
                numeric_values = self.numeric_scaler_.transform(numeric_values)
            transformed_blocks.append(numeric_values)

        if self.boolean_indices_:
            boolean_values = X_imputed[:, self.boolean_indices_].astype(int)
            transformed_blocks.append(boolean_values)

        for idx in self.categorical_indices_:
            encoder = self.encoders_[idx]
            encoded = encoder.transform(X_imputed[:, idx])
            if encoded.ndim == 1:
                encoded = encoded.reshape(-1, 1)
            transformed_blocks.append(encoded.astype(float))

        if not transformed_blocks:
            return np.empty((X.shape[0], 0), dtype=float)

        return np.hstack(transformed_blocks).astype(float)

    def fit_transform(self, X, feature_names=None):
        return self.fit(X, feature_names=feature_names).transform(X)

    def get_metadata(self):
        if self.metadata_ is None:
            raise RuntimeError("PreprocessingPipeline is not fitted yet. Call fit() first.")
        return self.metadata_
