import numpy as np
from base_model import BaseModel


class KNearestNeighbors(BaseModel):

    def __init__(self, k=5, distance_metric="euclidean"):
        """
        Parameters
        ----------
        k : int
            Number of neighbors
        distance_metric : str
            "euclidean" or "manhattan"
        """
        self.k = k
        self.distance_metric = distance_metric


    def fit(self, X, y):
        """
        KNN training simply stores the dataset.
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)


    def _distance(self, x1, x2):
        """
        Compute distance between two vectors.
        """

        if self.distance_metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))

        elif self.distance_metric == "manhattan":
            return np.sum(np.abs(x1 - x2))

        else:
            raise ValueError("Unsupported distance metric")


    def _predict_single(self, x):
        """
        Predict the label for a single sample.
        """

        distances = []

        for i in range(len(self.X_train)):

            d = self._distance(x, self.X_train[i])
            distances.append((d, self.y_train[i]))

        # sort neighbors by distance
        distances.sort(key=lambda item: item[0])

        # take the k closest neighbors
        neighbors = distances[:self.k]

        # extract labels
        neighbor_labels = np.array([label for _, label in neighbors])

        # majority vote using numpy
        labels, counts = np.unique(neighbor_labels, return_counts=True)

        return labels[np.argmax(counts)]


    def predict(self, X):
        """
        Predict labels for multiple samples.
        """

        X = np.array(X)

        predictions = []

        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)

        return np.array(predictions)