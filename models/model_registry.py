from models.KNN import KNearestNeighbors
from models.decision_tree import DecisionTree
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.naive_bayes import GaussianNaiveBayes
from models.random_forest import RandomForest


def get_model_registry():
    return {
        "classification": {
<<<<<<< HEAD
            "logistic_regression": LogisticRegression,
            #"decision_tree": DecisionTree,
            #"random_forest": RandomForest,
            #"naive_bayes": GaussianNaiveBayes,
            #"knn": KNearestNeighbors,
=======
           #"logistic_regression": LogisticRegression,
            #"decision_tree": DecisionTree,
            #"random_forest": RandomForest,
            "naive_bayes": GaussianNaiveBayes,
           # "knn": KNearestNeighbors,
>>>>>>> a9930c2 (b)
        },
        "regression": {
            "linear_regression": LinearRegression,
            #"decision_tree": DecisionTree,
            #"random_forest": RandomForest,
            #"knn": KNearestNeighbors,
        },
    }


def get_models_for_task(task):
    registry = get_model_registry()
    if task not in registry:
        raise ValueError(f"Unsupported task '{task}'. Expected one of {tuple(registry)}")
    return registry[task]
