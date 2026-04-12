class GlassBoxAutoML:
    """
    Orchestrator for AutoML pipeline.
    Responsibilities:
      - Receive cleaned data
      - Run hyperparameter search (Grid/Random)
      - Use K-Fold CV to evaluate models
      - Return best hyperparameters and score
    """
    def __init__(self, search_strategy, cross_validator):
        self.search_strategy = search_strategy  # instance of GridSearch or RandomSearch
        self.cross_validator = cross_validator  # instance of KFoldCV

    def run(self, X_clean, y_clean):
        """
        Executes the AutoML pipeline:
          1. Hyperparameter search with K-Fold CV
          2. Returns best score and parameters
        """
        best_score, best_params = self.search_strategy.search(X_clean, y_clean, self.cross_validator)
        return best_score, best_params