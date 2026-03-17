# Label Encoder : converts a categorical column into integer codes
# fit : learns the unique categories and assigns each an integer [only for training data]
# transform : replaces each category with its integer code [transform any data, test or train]
# unseen categories during transform, i assigned -1 (unknown)

import numpy as np


class LabelEncoder:
    def __init__(self):
        self.classes_ = None      # unique categories learned during fit
        self.mapping_ = None      # dict : category → integer code

    def fit(self, X):
        # X is a 1D ndarray of categorical values
        # learns unique categories and maps them to integers (alphabetical order)
        unique = sorted(set(X))
        self.classes_ = np.array(unique)
        self.mapping_ = {category: i for i, category in enumerate(unique)}
        return self

    def transform(self, X):
        # replaces each category with its learned integer code
        # unseen categories → -1
        if self.mapping_ is None:
            raise RuntimeError("LabelEncoder is not fitted yet. Call fit() first.")
        return np.array([self.mapping_.get(v, -1) for v in X])

    def fit_transform(self, X):
        return self.fit(X).transform(X)