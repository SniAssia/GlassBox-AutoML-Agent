import numpy as np

# Import des composants
from eda.auto_typing import AutoTyping
from eda.statistics import EDA 
from cleaner.simple_imputer import SimpleImputer
from cleaner.oneHotEncoder import OneHotEncoder

class GlassBoxAutoML:
    def __init__(self, search_strategy, cross_validator):
        self.search_strategy = search_strategy  
        self.cross_validator = cross_validator  
        self.best_model = None
        self.best_params = None
        self.best_score = None
        self.classes_ = None 
        self.report_metadata = {}

    def fit_end_to_end(self, raw_data, target_index, task='classification'):
        """
        Complete Pipeline: AutoTyping -> EDA -> Clean -> Search
        """
        # 1. INITIAL ANALYSIS
        at = AutoTyping()
        col_types = at.fit_transform(raw_data)
        
        eda = EDA()
        eda_report = eda.analyze(raw_data, col_types)
        self.report_metadata['eda'] = eda_report

        # 2. SEPARATE FEATURES AND TARGET
        y = raw_data[:, target_index]
        X = np.delete(raw_data, target_index, axis=1)
        feature_types = np.delete(col_types, target_index)

        # 3. PREPROCESSING (The "Clean" Stage)
        imputer = SimpleImputer(method='mean')
        X_imputed = imputer.fit_transform(X, feature_types)
        
        encoder = OneHotEncoder()
        X_final = encoder.fit_transform(X_imputed, feature_types)

        # --- AJOUT ICI : STANDARDIZATION (SCALING) ---
        # On calcule la moyenne et l'écart-type de chaque colonne
        means = np.mean(X_final, axis=0)
        stds = np.std(X_final, axis=0)
        
        # Sécurité : on remplace les std nuls par 1 pour éviter division par 0
        stds[stds == 0] = 1
        
        # On applique la formule (x - mean) / std
        X_scaled = (X_final - means) / stds
        # ---------------------------------------------

        # 4. SEARCH (The "Auto" Stage)
        print(f"Starting search using {type(self.search_strategy).__name__}...")
        
        unique_labels = np.unique(y)
        self.classes_ = unique_labels
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        y_numeric = np.array([label_to_idx[l] for l in y])

        # --- MODIFICATION ICI : On passe X_scaled au lieu de X_final ---
        self.best_score, self.best_params = self.search_strategy.search(
            X_scaled, y_numeric, self.cross_validator
        )
        
        self.best_model = self.search_strategy.best_model_instance 

        return self.generate_final_report()

    def generate_final_report(self):
        """
        Prepares the JSON-ready dictionary for the IronClaw Agent.
        """
        return {
            "status": "success",
            "score": round(float(self.best_score), 4) if self.best_score is not None else 0.0,
            "hyperparameters": self.best_params,
            "model_type": type(self.best_model).__name__ if self.best_model else "Unknown",
            "classes_found": self.classes_.tolist() if self.classes_ is not None else [],
            "eda_summary": self.report_metadata['eda']
        }