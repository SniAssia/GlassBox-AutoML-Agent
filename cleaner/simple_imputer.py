import numpy as np
from eda.statistics import median, mean, stdev, mode


class SimpleImputer:

    def __init__(self, method = 'mean'):
        # mathod is 'mean' or 'median' if column is numeric 
        # categorical columns are imputed with 'mode' (most frequent category)

        if method not in ('mean', 'median'):
            raise ValueError("method must be either 'mean' or 'median'")
        self.method_ = method
        self.fill_values_ = None

    def _is_missing(self,col):
        """this returns a boolean array of the same size as col. It indicates with True where missings values are, i.e np.nan or None"""
        return np.array([v is None or(isinstance(v,float) and np.isnan(v)) for v in col])
    
    def _is_numeric(self, col):
        """this removes missing values from the column, then tries to cast it to float. 
            If it succeeds → numerical. If it throws an error (e.g. column contains strings like "cat") => categorical."""
        try: 
            col_clean = col[~self._is_missing(col)] # It contains only values that were false when applying _is_mising
            col_clean.astype(float)
            return True
        
        except (ValueError,TypeError):
            return False
        
    def fit(self, X):
        # learns one fill value per column from training data
        # numerical columns → mean or median depending on self.method
        # categorical columns → mode always
        # stores results in fill_values_ (one value per column)
        self.fill_values_ = []
        for i in range(X.shape[1]):
            col = X[:, i]
            missing_mask = self._is_missing(col)
            if self._is_numeric(col):
                col_clean = col[~missing_mask].astype(float)
                if self.method_ == 'mean':
                    self.fill_values_.append(mean(col_clean))
                elif self.method_ == 'median':
                    self.fill_values_.append(median(col_clean))
            else:
                col_clean = col[~missing_mask]
                self.fill_values_.append(mode(col_clean))
        return self
    
    def transform(self, X):
        # fills missing values (np.nan or None) in each column using the stored fill_values_
        # works on a copy of X to avoid modifying the original data
        if self.fill_values_ is None:
            raise RuntimeError("SimpleImputer is not fitted yet. Call fit() first.")
        X_out = X.copy()
        for i in range(X_out.shape[1]):
            col = X_out[:, i]
            missing_mask = self._is_missing(col)
            X_out[missing_mask, i] = self.fill_values_[i]
        return X_out


    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)

    def fit_Transform(self, X_train):
        return self.fit_transform(X_train)