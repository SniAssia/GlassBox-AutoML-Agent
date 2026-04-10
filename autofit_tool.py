import json
import base64
from utils.csv_parser import parse_csv_bytes
from Orchestrator.Orchestrator import GlassBoxAutoML
from Orchestrator.RandomSearch import RandomSearch 
from Orchestrator.KFoldCV import KFoldCV

# --- IMPORT DE TOUS TES MODÈLES ---
from models.logistic_regression import LogisticRegression 
from models.decision_tree import DecisionTree
from models.random_forest import RandomForest
from models.naive_bayes import GaussianNaiveBayes
from models.KNN import KNearestNeighbors

def run_autofit(csv_base64, target_column, task_type='classification'):
    try:
        csv_bytes = base64.b64decode(csv_base64)
        data, headers = parse_csv_bytes(csv_bytes)
        target_index = headers.index(target_column)

        # 1. DÉFINITION DU ZOO DE MODÈLES AVEC FILTRE DE TÂCHE
        all_configs = [
            {
                "class": LogisticRegression,
                "tasks": ["classification"],
                "params": {'lr': [0.01, 0.1, 0.5], 'n_epochs': [100, 500]}
            },
            {
                "class": DecisionTree,
                "tasks": ["classification", "regression"],
                "params": {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]}
            },
            {
                "class": RandomForest,
                "tasks": ["classification", "regression"],
                "params": {
                    "n_trees": [10, 50],
                    "max_depth": [5, 10, 20],
                    "min_samples_split": [2, 5]
                }
            },
            {
                "class": GaussianNaiveBayes,
                "tasks": ["classification"],
                "params": {}
            },
            {
                "class": KNearestNeighbors,
                "tasks": ["classification", "regression"],
                "params": {'k': [3, 5, 7]}
            }
        ]

        # --- FILTRAGE AUTOMATIQUE ---
        # On ne garde que les modèles compatibles avec la tâche demandée
        model_configs = [m for m in all_configs if task_type in m["tasks"]]

        if not model_configs:
            return json.dumps({"status": "failed", "error": f"Aucun modèle disponible pour la tâche: {task_type}"})

        best_overall_results = None
        max_score = -1.0

        # 2. LA BOUCLE DE COMPÉTITION
        for config in model_configs:
            print(f"--- Évaluation de : {config['class'].__name__} (Task: {task_type}) ---")
            
            search_strategy = RandomSearch(
                model_class=config['class'], 
                param_distributions=config['params'],
                n_iter=3 
            ) 
            
            cv = KFoldCV(n_splits=3)
            automl = GlassBoxAutoML(search_strategy, cv)

            # L'orchestrateur reçoit la tâche pour adapter les calculs (Gini vs MSE)
            current_results = automl.fit_end_to_end(data, target_index, task=task_type)
            
            # 3. MISE À JOUR DU CHAMPION
            if current_results['score'] > max_score:
                max_score = current_results['score']
                best_overall_results = current_results

        return json.dumps(best_overall_results, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({"status": "failed", "error": str(e), "trace": traceback.format_exc()})