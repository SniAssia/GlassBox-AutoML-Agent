import numpy as np
from collections import Counter
from models.base_model import BaseModel
from models.decision_tree import DecisionTree
class RandomForest(BaseModel):
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            # 1. Create a Bootstrap sample (Random rows with replacement)
            X_sample, y_sample = self._bootstrap_samples(X, y)
            
            # 2. Initialize your existing DecisionTree
            tree = DecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split,
                n_features=self.n_features # Tell tree to only look at some features
            )
            
            # 3. Train the tree on the random sample
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_samples(self, X, y): # X features matrix with rows filled 
        n_samples = X.shape[0] # num of rows 
        # Pick random indices (with replacement)
        #index positions of the data I want to grab.
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def predict(self, X): # X matrix of input rows to predict 
        # 1. Get predictions from every single tree
        # tree_preds shape: (n_trees, n_samples)
        #Each Row of this matrix represents one tree's opinion on all 100 samples.
        #Each Column represents all trees' opinions on a single sample.
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        
        # 2. Let the trees vote! 
        # We swap axes to loop through each sample's set of predictions
        #By swapping the axes, you align the data so that each row represents one individual's set of votes.
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        # 3. Pick the most common label (Majority Vote)
        predictions = [Counter(sample_preds).most_common(1)[0][0] for sample_preds in tree_preds]
        return np.array(predictions)