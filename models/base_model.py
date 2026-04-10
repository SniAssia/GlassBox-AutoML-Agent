from abc import ABC, abstractmethod
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all models in the Algorithm Zoo.
    Every model must implement fit() and predict().
    """

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model.

        Parameters
        ----------
        X : array-like
            Training features
        y : array-like
            Target labels
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict outputs for input samples.

        Parameters
        ----------
        X : array-like
            Input features

        Returns
        -------
        predictions : array-like
        """
        pass
    def score(self, X, y):
        task = getattr(self, "task", None)
        if task is None:
            raise ValueError("Model must define self.task as 'classification' or 'regression'")
        y_pred = self.predict(X)

        if task == "classification":
            return np.mean(y_pred == y)  # accuracy

        elif task == "regression":
            return -np.mean((y - y_pred) ** 2)  # negative MSE for max-based search

        else:
            raise ValueError("Unsupported task type")