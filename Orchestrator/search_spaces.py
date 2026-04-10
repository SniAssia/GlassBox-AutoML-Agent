def get_search_spaces(task):
    spaces = {
        "classification": {
            "logistic_regression": {
                "grid": {
                    "lr": [0.01, 0.05],
                    "n_epochs": [300, 800],
                    "tol": [1e-4, 1e-6],
                },
                "random": {
                    "lr": [0.001, 0.01, 0.05, 0.1],
                    "n_epochs": [200, 400, 800, 1200],
                    "tol": [1e-3, 1e-4, 1e-5, 1e-6],
                },
            },
            "decision_tree": {
                "grid": {
                    "max_depth": [3, 5, 10],
                    "min_samples_split": [2, 4, 8],
                    "n_features": [None],
                },
                "random": {
                    "max_depth": [3, 4, 5, 8, 10, 15],
                    "min_samples_split": [2, 3, 4, 6, 8],
                    "n_features": [None],
                },
            },
            "random_forest": {
                "grid": {
                    "n_trees": [5, 10],
                    "max_depth": [5, 10],
                    "min_samples_split": [2, 4],
                    "n_features": [None],
                },
                "random": {
                    "n_trees": [5, 10, 20, 30],
                    "max_depth": [4, 6, 8, 10, 15],
                    "min_samples_split": [2, 3, 4, 6],
                    "n_features": [None],
                },
            },
            "naive_bayes": {
                "grid": {
                    "var_smoothing": [1e-9, 1e-8, 1e-7],
                },
                "random": {
                    "var_smoothing": [1e-10, 1e-9, 1e-8, 1e-7, 1e-6],
                },
            },
            "knn": {
                "grid": {
                    "k": [3, 5, 7],
                    "distance_metric": ["euclidean", "manhattan"],
                },
                "random": {
                    "k": [1, 3, 5, 7, 9, 11],
                    "distance_metric": ["euclidean", "manhattan"],
                },
            },
        },
        "regression": {
            "linear_regression": {
                "grid": {
                    "solver": ["gd", "normal"],
                    "lr": [0.001, 0.01],
                    "n_epochs": [300, 800],
                    "tol": [1e-4, 1e-6],
                },
                "random": {
                    "solver": ["gd", "normal"],
                    "lr": [0.0005, 0.001, 0.01, 0.05],
                    "n_epochs": [200, 400, 800, 1200],
                    "tol": [1e-3, 1e-4, 1e-5, 1e-6],
                },
            },
            "decision_tree": {
                "grid": {
                    "max_depth": [3, 5, 10],
                    "min_samples_split": [2, 4, 8],
                    "n_features": [None],
                },
                "random": {
                    "max_depth": [3, 4, 5, 8, 10, 15],
                    "min_samples_split": [2, 3, 4, 6, 8],
                    "n_features": [None],
                },
            },
            "random_forest": {
                "grid": {
                    "n_trees": [5, 10],
                    "max_depth": [5, 10],
                    "min_samples_split": [2, 4],
                    "n_features": [None],
                },
                "random": {
                    "n_trees": [5, 10, 20, 30],
                    "max_depth": [4, 6, 8, 10, 15],
                    "min_samples_split": [2, 3, 4, 6],
                    "n_features": [None],
                },
            },
            "knn": {
                "grid": {
                    "k": [3, 5, 7],
                    "distance_metric": ["euclidean", "manhattan"],
                },
                "random": {
                    "k": [1, 3, 5, 7, 9, 11],
                    "distance_metric": ["euclidean", "manhattan"],
                },
            },
        },
    }
    if task not in spaces:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {tuple(spaces)}")
    return spaces[task]


def get_model_search_config(task, model_name, search_strategy="grid"):
    task_spaces = get_search_spaces(task)
    if model_name not in task_spaces:
        raise ValueError(f"No search space configured for model '{model_name}' in task '{task}'")
    if search_strategy not in task_spaces[model_name]:
        raise ValueError(f"Unknown search strategy '{search_strategy}'")
    return task_spaces[model_name][search_strategy]
