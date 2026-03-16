class HyperparameterSearch:
    """
    Base class for hyperparameter search strategies.
    Subclasses should implement the search() method.
    """
    def __init__(self, model_factory):
        self.model_factory = model_factory  # creates models with given hyperparameters

    def search(self, X, y, cv):
        raise NotImplementedError("Subclasses must implement this method")