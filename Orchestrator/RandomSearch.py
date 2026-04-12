import numpy as np

class RandomSearch:
    def __init__(self, model_class, param_distributions, n_iter=10, seed=42):
        self.model_class = model_class
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.seed = seed

        if isinstance(model_class, type):
            self._ctor = model_class
        else:
            self._ctor = model_class.__class__

    def _sample_params(self):
        """
        Sample one set of hyperparameters randomly using only numpy
        """
        sampled = {}
        for k, v in self.param_distributions.items():
            arr = np.array(v)
            idx = np.random.randint(0, len(arr))
            sampled[k] = arr[idx]
        return sampled

    def search(self, X, y, cv):
        best_score = -np.inf
        best_params = None
        np.random.seed(self.seed)
        for _ in range(self.n_iter):
            params = self._sample_params()
            scores = []
            for train_idx, val_idx in cv.split(X):
                model = self._ctor(**params)
                model.fit(X[train_idx], y[train_idx])
                score = model.score(X[val_idx], y[val_idx])
                scores.append(score)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        return best_score, best_params