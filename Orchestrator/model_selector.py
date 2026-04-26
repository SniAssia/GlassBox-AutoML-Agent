import inspect

import numpy as np

from Orchestrator.GridSearch import GridSearch
from Orchestrator.KFoldCV import KFoldCV
from Orchestrator.RandomSearch import RandomSearch
from Orchestrator.search_spaces import get_model_search_config
from models.model_registry import get_models_for_task


def _make_model_factory(model_class, task):
    signature = inspect.signature(model_class.__init__)
    supported_params = set(signature.parameters)

    def factory(**params):
        instance_params = dict(params)
        if "task" in supported_params:
            instance_params["task"] = task
        return model_class(**instance_params)

    return factory


def _to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


class ModelSelector:
    def __init__(self, task, search_strategy="grid", cv_splits=5, random_iter=10, seed=42):
        if search_strategy not in ("grid", "random"):
            raise ValueError("search_strategy must be 'grid' or 'random'")

        self.task = task
        self.search_strategy = search_strategy
        self.cv_splits = cv_splits
        self.random_iter = random_iter
        self.seed = seed
        self.cv = KFoldCV(n_splits=cv_splits, shuffle=True, seed=seed)

    def _make_search(self, factory, model_name):
        params = get_model_search_config(self.task, model_name, self.search_strategy)
        if self.search_strategy == "grid":
            return GridSearch(factory, params)
        return RandomSearch(factory, params, n_iter=self.random_iter, seed=self.seed)

    def select(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        best_result = None
        tried_models = []
        models = get_models_for_task(self.task)

        for model_name, model_class in models.items():
            factory = _make_model_factory(model_class, self.task)
            try:
                search = self._make_search(factory, model_name)
                best_score, best_params = search.search(X, y, self.cv)
                best_params = {key: _to_builtin(value) for key, value in (best_params or {}).items()}

                fitted_model = factory(**best_params)
                fitted_model.fit(X, y)

                result = {
                    "model_name": model_name,
                    "cv_score": float(best_score),
                    "best_params": best_params,
                    "fitted_model": fitted_model,
                }
                tried_models.append(
                    {
                        "model_name": model_name,
                        "cv_score": float(best_score),
                        "best_params": best_params,
                    }
                )

                if best_result is None or result["cv_score"] > best_result["cv_score"]:
                    best_result = result
            except Exception as exc:
                tried_models.append(
                    {
                        "model_name": model_name,
                        "error": str(exc),
                    }
                )
                continue

        if best_result is None:
            raise ValueError(f"No model could be trained successfully for task '{self.task}'.")

        return {
            "task": self.task,
            "search_strategy": self.search_strategy,
            "cv_splits": self.cv_splits,
            "tried_models": tried_models,
            "best_model_name": best_result["model_name"],
            "best_score": best_result["cv_score"],
            "best_params": best_result["best_params"],
            "best_model": best_result["fitted_model"],
        }
