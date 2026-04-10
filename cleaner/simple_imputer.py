import numpy as np
from eda.statistics import EDA  # On importe la classe EDA

class SimpleImputer:
    def __init__(self, method='mean'):
        if method not in ('mean', 'median'):
            raise ValueError("method must be either 'mean' or 'median'")
        self.method_ = method
        self.fill_values_ = None
        self.eda = EDA()

    def _is_missing(self, col):
        return np.array([
            v is None or 
            (isinstance(v, float) and np.isnan(v)) or 
            (isinstance(v, str) and str(v).strip() == "") 
            for v in col
        ])
    
    # AJOUT de col_types ici pour correspondre à l'appel de l'Orchestrateur
    def fit(self, X, col_types=None):
        self.fill_values_ = []
        X = np.asarray(X)
        
        for i in range(X.shape[1]):
            col = X[:, i]
            missing_mask = self._is_missing(col)
            col_clean = col[~missing_mask]

            if col_clean.size == 0:
                self.fill_values_.append(0)
                continue

            # On utilise col_types s'il est fourni, sinon on devine
            is_num = False
            if col_types is not None:
                is_num = (col_types[i] == "numerical")
            else:
                try:
                    col_clean.astype(float)
                    is_num = True
                except:
                    is_num = False

            if is_num:
                col_numeric = col_clean.astype(float)
                if self.method_ == 'mean':
                    self.fill_values_.append(self.eda.mean(col_numeric))
                elif self.method_ == 'median':
                    self.fill_values_.append(self.eda.median(col_numeric))
            else:
                self.fill_values_.append(self.eda.mode(col_clean))
        return self
    
    def transform(self, X):
        if self.fill_values_ is None:
            raise RuntimeError("SimpleImputer is not fitted yet. Call fit() first.")
        X_out = np.copy(X)
        for i in range(X_out.shape[1]):
            missing_mask = self._is_missing(X_out[:, i])
            X_out[missing_mask, i] = self.fill_values_[i]
        return X_out

    # AJOUT de col_types ici aussi
    def fit_transform(self, X, col_types=None):
        return self.fit(X, col_types).transform(X)