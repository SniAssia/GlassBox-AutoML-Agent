# One Hot Encoder : converts a categorical column into a binary matrix
# fit : learns the unique categories and creates one column per category [only for training data]
# transform : returns a binary matrix — 1 where the category matches, 0 elsewhere [transform any data, test or train]
# unseen categories during transform => a separate 'unknown' column fires instead

import numpy as np


class OneHotEncoder:
    def __init__(self):
        self.classes_ = None      # unique categories learned during fit
        self.mapping_ = None      # dict : category → column index

    def fit(self, X):
        # X is a 1D ndarray of categorical values
        # learns unique categories and assigns each a column index
        unique = sorted(set(X))
        self.classes_ = np.array(unique)
        self.mapping_ = {category: i for i, category in enumerate(unique)}
        return self

    def transform(self, X):
        # returns a binary matrix of shape (n_samples, n_categories + 1)
        # last column is the 'unknown' column — fires for unseen categories
        if self.mapping_ is None:
            raise RuntimeError("OneHotEncoder is not fitted yet. Call fit() first.")

        n_samples = len(X)
        n_cols = len(self.classes_) + 1          # +1 for the unknown column
        out = np.zeros((n_samples, n_cols), dtype=int)

        for i, val in enumerate(X):
            if val in self.mapping_:
                out[i, self.mapping_[val]] = 1   # known category → fire its column
            else:
                out[i, -1] = 1                   # unseen category → fire unknown column

        return out

    def fit_transform(self, X):
        return self.fit(X).transform(X)