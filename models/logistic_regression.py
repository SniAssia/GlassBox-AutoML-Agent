# Logistic Regression : multi-class classification using One-vs-Rest (OvR) strategy
# Each class gets its own binary classifier trained with gradient descent
# fit         : trains K binary classifiers (one per class) using cross-entropy loss
# predict_proba : returns probability estimates for each class
# predict       : returns the class with the highest probability

import numpy as np
from models.base_model import BaseModel


class LogisticRegression(BaseModel):
    def __init__(self, lr=0.01, n_epochs=1000, tol=1e-6, threshold=0.5):
        # lr        : learning rate (constant)
        # n_epochs  : maximum number of gradient descent iterations
        # tol       : early stopping — stop if cost improvement < tol
        # threshold : decision boundary for binary prediction (used in OvR voting)
        self.lr        = lr
        self.n_epochs  = n_epochs
        self.tol       = tol
        self.threshold = threshold
        self.weights_  = None   # shape (K, n+1) — one weight vector per class
        self.classes_  = None   # unique class labels learned during fit

    def _add_bias(self, X):
        # prepends a column of ones to absorb bias into w (same as LinearRegression)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _sigmoid(self, z):
        # sigmoid : maps any real value to (0, 1)
        # clipped to avoid overflow in exp for very large/small z
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _cost(self, X, y_binary, w):
        # binary cross-entropy loss derived from maximum likelihood estimation
        m = X.shape[0]
        p = self._sigmoid(X @ w)
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -(1 / m) * np.sum(y_binary * np.log(p) + (1 - y_binary) * np.log(1 - p))

    def _gradient(self, X, y_binary, w):
        # gradient of cross-entropy : (1/m) * X.T @ (sigmoid(Xw) - y)
        # the p(1-p) from sigmoid derivative cancels with the denominator in cost derivative
        m = X.shape[0]
        errors = self._sigmoid(X @ w) - y_binary
        return (1 / m) * (X.T @ errors)

    def _fit_binary(self, X, y_binary):
        # trains one binary classifier for one class using gradient descent
        # y_binary : 1 where sample belongs to this class, 0 elsewhere
        w = np.zeros(X.shape[1])
        costs = []

        for epoch in range(self.n_epochs):
            cost = self._cost(X, y_binary, w)
            costs.append(cost)

            # early stopping — stop if cost improvement is smaller than tol
            if epoch > 0 and abs(costs[-2] - costs[-1]) < self.tol:
                break

            w = w - self.lr * self._gradient(X, y_binary, w)

        return w

    def fit(self, X, y):
        # X : 2D numerical ndarray of shape (m, n)
        # y : 1D array of class labels of shape (m,) — any integer or string labels
        X = self._add_bias(X)
        self.classes_ = np.unique(y)
        K = len(self.classes_)

        # train one binary classifier per class (One-vs-Rest)
        self.weights_ = np.zeros((K, X.shape[1]))
        for i, cls in enumerate(self.classes_):
            y_binary = (y == cls).astype(float)   # 1 for this class, 0 for all others
            self.weights_[i] = self._fit_binary(X, y_binary)

        return self

    def predict_proba(self, X):
        # returns probability matrix of shape (m, K)
        # normalized so each row sums to 1
        if self.weights_ is None:
            raise RuntimeError("LogisticRegression is not fitted yet. Call fit() first.")

        X = self._add_bias(X)
        probs = np.array([self._sigmoid(X @ w) for w in self.weights_]).T  # shape (m, K)

        # normalize rows — OvR probabilities don't naturally sum to 1
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / row_sums

    def predict(self, X):
        # returns predicted class label — class with highest probability wins
        if self.weights_ is None:
            raise RuntimeError("LogisticRegression is not fitted yet. Call fit() first.")

        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]