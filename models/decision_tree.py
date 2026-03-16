from models.base_model import BaseModel
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, prediction=None, samples_count=0, impurity=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction # if this value is None, it means it is a decision node
        self.samples_count = samples_count # counts how many samples reached the node
        self.impurity = impurity # counts how pure this node is



class DecisionTree(BaseModel):
    def __init__(self):
        pass

    def gini_impurity(self, y): # is a 1D array of class labels
        n = len(y)
        if n == 0: 
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        counts = counts / n
        return 1 - np.sum(counts ** 2)
    
    def mse_impurity(self, y):
        y = np.asarray(y, dtype=float)
        n = len(y)
        if n == 0:
            return 0.0
        mu = np.mean(y)
        return np.sum((y - mu) ** 2) / n
    
