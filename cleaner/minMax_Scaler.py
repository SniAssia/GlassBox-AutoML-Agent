# MinMax scaler : is (z - x_min) / (x_max - x_min)
# fit is just calculating min and max for later scaling (i bring this from eda package again) [only for training data]
# transform is just applying the transformation to a matrix -> i return a transformed array [transform any data, test or train]


# minMax scaler is very sensitive in the presence of outliers
import numpy as np
from eda.statistics import min_val, max_val


class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X):
        # X is assumed to be a numerical ndarray — encoding is handled upstream by the agent
        self.min_ = np.array([min_val(X[:, i]) for i in range(X.shape[1])])
        self.max_ = np.array([max_val(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self, X):
        if self.min_ is None:
            raise RuntimeError("MinMaxScaler is not fitted yet. Call fit() first.")

        # avoid division by zero for constant columns (x_max == x_min). in this case I default to range = 1.
        range_safe = np.where((self.max_ - self.min_) == 0, 1, self.max_ - self.min_)
        return (X - self.min_) / range_safe

    def fit_transform(self, X_train):
        return self.fit(X_train).transform(X_train)