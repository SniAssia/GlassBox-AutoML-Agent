from abc import ABC, abstractmethod


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