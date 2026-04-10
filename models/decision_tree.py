from models.base_model import BaseModel
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None, samples_count=0, impurity=0.0):
        self.feature = feature
        self.threshold = threshold # used for numeric features to decide how to split the data (e.g. feature <= threshold goes left, else goes right)
        self.left = left
        self.right = right
        self.prediction = prediction # if this value is None, it means it is a decision node
        self.samples_count = samples_count # counts how many samples reached the node
        self.impurity = impurity # counts how pure this node is

class DecisionTree(BaseModel):
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None, task='classification'):
        # On vérifie d'abord si c'est None avant de comparer
        if max_depth is not None and max_depth < 1:
            raise ValueError('max_depth must be >= 1')
            
        if min_samples_split < 2:
            raise ValueError('min_samples_split must be >= 2')
        # ... reste du init ...
        if task not in ('classification', 'regression'):
            raise ValueError("task must be 'classification' or 'regression'")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.task = task
        self.root = None
        self._resolved_n_features = None

    def gini_impurity(self, y): # is a 1D array of class labels
        n = len(y)
        if n == 0: 
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        counts = counts / n
        return 1 - np.sum(counts ** 2)
    
    def mse_impurity(self, y):
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            return 0.0
        mu = np.mean(y)
        return np.sum((y - mu) ** 2) / n
    
    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError('X must be a 2D array')
        if y.ndim != 1:
            raise ValueError('y must be a 1D array')
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of samples')

        if self.n_features is None:
            self._resolved_n_features = X.shape[1] # num of columns (features)
        else:
            self._resolved_n_features = min(self.n_features, X.shape[1])
        if self._resolved_n_features < 1:
            raise ValueError('n_features must be >= 1')
        
        self.root = self._grow_tree(X, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # Dans models/decision_tree.py

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        is_pure = self._is_pure(y)

        # --- MODIFICATION ICI ---
        # On vérifie si max_depth est atteint SEULEMENT s'il n'est pas None
        depth_reached = (self.max_depth is not None and depth >= self.max_depth)
        
        if (depth_reached or 
            n_samples < self.min_samples_split or 
            is_pure):
            leaf_value = self._leaf_value(y)
            return Node(prediction=leaf_value, samples_count=n_samples)
        # -------------------------

        # Randomly select features to consider for splitting
        feature_indices = np.random.choice(n_features, self._resolved_n_features, replace=False)
        best_feature, best_threshold = self._best_split(X, y, feature_indices)
        if best_feature is None:
            leaf_value = self._leaf_value(y)
            return Node(prediction=leaf_value, samples_count=n_samples)
        # Split the dataset
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = X[:, best_feature] > best_threshold
        left_child = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow_tree(X[right_indices], y[right_indices], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child, samples_count=n_samples)
        
    def _best_split(self, X, y, feature_indices):
        best_gain = -np.inf
        best_feature, best_threshold = None, None
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_idx, right_idx = self._split(X[:, feature], threshold)
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                gain = self._information_gain(y, left_idx, right_idx)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def _split(self, X_column, threshold):
        left_indices = np.where(X_column <= threshold)[0]
        right_indices = np.where(X_column > threshold)[0]
        return left_indices, right_indices
    
    def _impurity(self, y):
        if self.task == 'classification':
            return self.gini_impurity(y)
        else:
            return self.mse_impurity(y)
        
    def _information_gain(self, y, left_idx, right_idx):
        parent_impurity = self._impurity(y)
        n = len(y)
        n_left, n_right = len(left_idx), len(right_idx)
        if n_left == 0 or n_right == 0:
            return 0
        left_impurity = self._impurity(y[left_idx])
        right_impurity = self._impurity(y[right_idx])
        child_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
        return parent_impurity - child_impurity
    
    def _leaf_value(self, y):
        if self.task == 'classification':
            return self._most_common_label(y)
        else:
            return np.mean(y)
        
    def _is_pure(self, y):
        if self.task == 'classification':
            return len(np.unique(y)) == 1
        return np.allclose(y, y[0])
    
    def _traverse_tree(self, x, node):
        if node.prediction is not None:
            return node.prediction
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        most_common = values[np.argmax(counts)]
        return most_common
