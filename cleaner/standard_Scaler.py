# Standard scaler : is (z - mean)/std 
# fit is just calculating mean and std for later scaling -> i return mean and std (i bring this from eda package) [only for training data]
# transform is just applying the transformation to an array-like or a matrix -> i return a transformed array [transform any data, test or train]

import numpy as np
from eda.statistics import stdev, mean 


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.stadev_ = None

    def fit(self, X):
        # X is assumed to be a numerical ndarray — encoding is handled upstream by the agent
        self.mean_   = np.array([mean(X[:, i])  for i in range(X.shape[1])])
        self.stadev_ = np.array([stdev(X[:, i]) for i in range(X.shape[1])])
        return self
    
    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("StandardScaler is not fitted yet. Call fit() first.")
        
        # avoid division by zero for constant columns (std == 0). in this case I default to std = 1.
        std_safe = np.where(self.stadev_ == 0, 1, self.stadev_)
        return (X - self.mean_) / std_safe

    def fit_transform(self, X):
        return self.fit(X).transform(X)