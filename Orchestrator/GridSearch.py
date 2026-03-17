import numpy as np

class GridSearch:
    def __init__(self, model_class, param_grid):
        """
        model_class: BaseModel subclass
        param_grid: dict, e.g., {'k': [3,5,7], 'metric': ['euclidean','manhattan']}
        """
        self.model_class = model_class
        self.param_grid = param_grid

    def _all_combinations(self):
        """
        Generate all combinations using only numpy
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        # Convert all values to arrays
        arrays = [np.array(v) for v in values]
        # Create meshgrid
        mesh = np.meshgrid(*arrays, indexing='ij')
        # Flatten and combine into list of dicts
        flat = [m.flatten() for m in mesh]
        combinations = []
        for i in range(flat[0].size):
            comb = {keys[j]: flat[j][i] for j in range(len(keys))}
            combinations.append(comb)
        return combinations

    def search(self, X, y, cv):
        best_score = -np.inf
        best_params = None
        for params in self._all_combinations():
            scores = []
            for train_idx, val_idx in cv.split(X):
                model = self.model_class(**params)
                model.fit(X[train_idx], y[train_idx])
                score = model.score(X[val_idx], y[val_idx])
                scores.append(score)
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
        return best_score, best_params