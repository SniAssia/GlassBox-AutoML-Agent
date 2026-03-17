import numpy as np
from eda.statistics import median, mean, stdev, mode


class SimpleImputer:

    def __init__(self, method = 'mean'):
        # mathod is 'mean' or 'median' if column is numeric 
        # categorical columns are imputed with 'mode' (most frequent category)

        self.method_ = mean
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
            pass

        def transform(self, X):
            pass

        def fit_Transform(self, X_train) :
            pass