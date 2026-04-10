import numpy as np

class AutoTyping:
    def __init__(self):
        pass

    def fit_transform(self, X):
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError("AutoTyping expects a 2D array")
        return [self.infer_type(X[:, i]) for i in range(X.shape[1])]

    def infer_type(self, x):
        if self.is_boolean(x):
            return "boolean"
        if self.is_numerical(x):
            return "numerical"
        return "categorical"

    def is_boolean(self, x):
        x = np.asarray(x)
        values = np.unique(x)
        # Gestion des strings qui pourraient représenter des booléens
        return np.all(np.isin(values, [0, 1, '0', '1', True, False, 'True', 'False']))

    def is_numerical(self, x):
        try:
            np.asarray(x, dtype=float)
            if self.is_boolean(x): return False
            return True
        except:
            return False