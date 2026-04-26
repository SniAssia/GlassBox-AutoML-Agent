import numpy as np


class RandomSearch:
    def __init__(self, model_class, param_distributions, n_iter=10, seed=42):
        self.model_class = model_class
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.seed = seed
        np.random.seed(seed)

    def _sample_params(self):
        """
        Sample one set of hyperparameters randomly using only numpy.
        """
        sampled = {}
        for key, values in self.param_distributions.items():
            arr = np.array(values)
            idx = np.random.randint(0, len(arr))
            sampled[key] = arr[idx]
        return sampled

    def search(self, X, y, cv):
        best_score = -np.inf
        best_params = None
        last_error = None
        for _ in range(self.n_iter):
            params = self._sample_params()
            scores = []
            try:
                for train_idx, val_idx in cv.split(X):
                    model = self.model_class(**params)
                    model.fit(X[train_idx], y[train_idx])
                    score = model.score(X[val_idx], y[val_idx])
                    scores.append(score)
            except Exception as exc:
                last_error = exc
                continue
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        if best_params is None:
            raise ValueError(f"RandomSearch could not evaluate any valid parameter combination: {last_error}")
        return best_score, best_params
