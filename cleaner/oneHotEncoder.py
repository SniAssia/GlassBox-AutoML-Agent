import numpy as np

class OneHotEncoder:
    def __init__(self):
        self.mapping_ = {}  # Stocke les catégories pour chaque index de colonne
        self.fitted_ = False

    def fit(self, X, col_types=None):
        """
        X: Matrice 2D de données.
        col_types: Liste des types ['numerical', 'categorical', ...]
        """
        X = np.asarray(X)
        self.mapping_ = {}
        
        # On parcourt chaque colonne
        for i in range(X.shape[1]):
            # On n'encode que si l'Orchestrateur dit que c'est du catégoriel
            if col_types is not None and col_types[i] == 'categorical':
                unique_vals = sorted(set(X[:, i]))
                self.mapping_[i] = {val: idx for idx, val in enumerate(unique_vals)}
        
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("OneHotEncoder is not fitted yet. Call fit() first.")
        
        X = np.asarray(X)
        n_samples = X.shape[0]
        final_cols = []

        for i in range(X.shape[1]):
            col_data = X[:, i]
            
            if i in self.mapping_:
                # Encodage de la colonne catégorielle
                n_cats = len(self.mapping_[i])
                # +1 pour la colonne 'unknown' (unseen categories)
                encoded = np.zeros((n_samples, n_cats + 1), dtype=int)
                
                for row_idx, val in enumerate(col_data):
                    if val in self.mapping_[i]:
                        encoded[row_idx, self.mapping_[i][val]] = 1
                    else:
                        encoded[row_idx, -1] = 1 # Colonne 'unknown'
                
                # On ajoute les colonnes encodées une par une
                for c in range(encoded.shape[1]):
                    final_cols.append(encoded[:, c])
            else:
                # Colonne numérique : on la garde telle quelle
                try:
                    final_cols.append(col_data.astype(float))
                except:
                    # Sécurité si une colonne numérique contient du texte
                    final_cols.append(np.zeros(n_samples))

        return np.column_stack(final_cols)

    def fit_transform(self, X, col_types=None):
        return self.fit(X, col_types).transform(X)