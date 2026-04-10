import numpy as np

class RandomSearch:
    def __init__(self, model_class, param_distributions, n_iter=10, seed=42):
        self.model_class = model_class
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.seed = seed
        self.best_model_instance = None  # <--- AJOUT : Initialisation
        np.random.seed(seed)

    def _sample_params(self):
        sampled = {}
        for k, v in self.param_distributions.items():
            arr = np.array(v)
            idx = np.random.randint(0, len(arr))
            # Utilisation de .item() pour convertir le type numpy en type Python standard
            # cela évite des erreurs de sérialisation JSON plus tard.
            val = arr[idx]
            sampled[k] = val.item() if hasattr(val, 'item') else val
        return sampled

    def search(self, X, y, cv):
        best_score = -np.inf
        best_params = None
        
        for _ in range(self.n_iter):
            params = self._sample_params()
            scores = []
            
            # On garde une trace des modèles entraînés sur les folds
            temp_models = [] 
            
            for train_idx, val_idx in cv.split(X):
                model = self.model_class(**params)
                model.fit(X[train_idx], y[train_idx])
                score = model.score(X[val_idx], y[val_idx])
                scores.append(score)
                temp_models.append(model)
            
            mean_score = np.mean(scores)
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                # AJOUT : On ré-entraîne le meilleur modèle sur toute la donnée X
                # ou on récupère le dernier modèle du fold. 
                # Le plus propre pour l'AutoML est de ré-entraîner sur tout l'ensemble :
                final_model = self.model_class(**params)
                final_model.fit(X, y)
                self.best_model_instance = final_model 
                
        return best_score, best_params