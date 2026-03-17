import numpy as np

class KFoldCV:
    """
    K-Fold Cross-Validation splitter.
    
    """
    def __init__(self, n_splits=5, shuffle=True, seed=42):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.seed = seed

    def split(self, X):
        n_samples = len(X)
        indices = np.arange(n_samples)
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(indices)

        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[:n_samples % self.n_splits] += 1

        current = 0
        for fold_size in fold_sizes:
            val_idx = indices[current:current+fold_size]
            train_idx = np.concatenate([indices[:current], indices[current+fold_size:]])
            yield train_idx, val_idx
            current += fold_size