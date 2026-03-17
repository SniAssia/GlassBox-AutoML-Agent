import numpy as np
from base_model import BaseModel


class KNearestNeighbors(BaseModel):
    #task = classification or regression

    def __init__(self, k=5, distance_metric="euclidean", task="classification"):
        self.k = k
        self.distance_metric = distance_metric
        self.task = task  


    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)


    def _distance(self, x1, x2):

        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))

        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))

        else:
            raise ValueError("Unsupported distance metric")


    def _predict_single(self, x):

        distances = []

        for i in range(len(self.X_train)):
            d = self._distance(x, self.X_train[i])
            distances.append((d, self.y_train[i]))

        # sort by distance
        distances.sort(key=lambda item: item[0])

        neighbors = distances[:self.k]
        neighbor_values = np.array([label for _, label in neighbors])

        if self.task == "classification":
            labels, counts = np.unique(neighbor_values, return_counts=True)
            return labels[np.argmax(counts)]

        elif self.task == "regression":
            return np.mean(neighbor_values)

        else:
            raise ValueError("Unsupported task type")


    def predict(self, X):

        X = np.array(X)
        predictions = []

        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)

        return np.array(predictions)