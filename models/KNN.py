import numpy as np
from models.base_model import BaseModel

class KNearestNeighbors(BaseModel):
    """K-Nearest Neighbors classifier/regressor with optimizations."""
    
    def __init__(self, k=5, distance_metric="euclidean", task="classification"):
        self.k = k
        self.distance_metric = distance_metric
        self.task = task
        # Pre-compute distance function for faster access
        self._distance_fn = self._get_distance_function()
    
    def fit(self, X, y):
        """Fit the model with training data."""
        self.X_train = np.asarray(X, dtype=np.float32)
        self.y_train = np.asarray(y)
        # Pre-cache training set size for faster loops
        self._train_size = len(self.X_train)
    
    def _get_distance_function(self):
        """Return the appropriate distance function as a closure."""
        if self.distance_metric == "euclidean":
            return lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.distance_metric == "manhattan":
            return lambda x1, x2: np.sum(np.abs(x1 - x2))
        else:
            raise ValueError("Unsupported distance metric")
    
    def _distance(self, x1, x2):
        """Calculate distance between two points (maintains API compatibility)."""
        return self._distance_fn(x1, x2)
    
    def _predict_single(self, x):
        """Predict for a single sample using optimized numpy operations."""
        # Vectorized distance computation instead of loop
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1)) \
            if self.distance_metric == "euclidean" \
            else np.sum(np.abs(self.X_train - x), axis=1)
        
        # Use argpartition for O(n) k-selection instead of full sort O(n log n)
        k_indices = np.argpartition(distances, min(self.k, len(distances) - 1))[:self.k]
        neighbor_values = self.y_train[k_indices]
        
        if self.task == "classification":
            labels, counts = np.unique(neighbor_values, return_counts=True)
            return labels[np.argmax(counts)]
        elif self.task == "regression":
            return np.mean(neighbor_values)
        else:
            raise ValueError("Unsupported task type")
    
    def predict(self, X):
        """Predict for multiple samples."""
        X = np.asarray(X, dtype=np.float32)
        predictions = np.empty(len(X), dtype=self.y_train.dtype)
        
        for i, x in enumerate(X):
            predictions[i] = self._predict_single(x)
        
        return predictions