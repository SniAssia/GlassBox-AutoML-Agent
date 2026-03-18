# Linear Regression : predicts a continuous target using f(x) = Xw (bias absorbed into w via column of ones)
# fit : learns weights w using gradient descent or normal equation
# predict : returns predictions for any X
# supports solver='gd' (gradient descent) or solver='normal' (closed form normal equation)

import numpy as np
from models.base_model import BaseModel


class LinearRegression(BaseModel):
    def __init__(self, solver='gd', lr=0.01, n_epochs=1000, tol=1e-6):
        # solver   : 'gd' for gradient descent, 'normal' for closed form
        # lr       : learning rate (constant) — only used when solver='gd' (equivalent to alpha in sklearn)
        # n_epochs : number of iterations — only used when solver='gd'
        # tol      : early stopping — stop if cost improvement < tol
        self.solver   = solver
        self.lr       = lr
        self.n_epochs = n_epochs
        self.tol      = tol     # this is my tolearance
        self.w_       = None    # learned weights (includes bias as w[0])
        self.costs_   = []      # cost history per epoch (useful for debugging)
        self.task     = 'regression'

    def _add_bias(self, X):
        # prepends a column of ones to X to absorb the bias into w
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _cost(self, X, y):
        # MSE cost : (1/2m) * ||Xw - y||^2
        m = X.shape[0]
        errors = X @ self.w_ - y
        return (1 / (2 * m)) * np.dot(errors, errors)

    def _gradient(self, X, y):
        # gradient of MSE : (1/m) * X.T @ (Xw - y)
        m = X.shape[0]
        errors = X @ self.w_ - y
        return (1 / m) * (X.T @ errors)

    def _fit_gd(self, X, y):
        # gradient descent with constant learning rate and early stopping
        self.w_ = np.zeros(X.shape[1])
        self.costs_ = []

        for epoch in range(self.n_epochs):
            cost = self._cost(X, y)
            self.costs_.append(cost)

            # early stopping — stop if cost improvement is smaller than tol
            if epoch > 0 and abs(self.costs_[-2] - self.costs_[-1]) < self.tol:
                print(f"Converged at epoch {epoch}")
                break

            self.w_ = self.w_ - self.lr * self._gradient(X, y)

    def _fit_normal(self, X, y):
        # closed form : w = (X.T @ X)^-1 @ X.T @ y
        self.w_ = np.linalg.inv(X.T @ X) @ X.T @ y

    def fit(self, X, y):
        # X : 2D numerical ndarray of shape (m, n)
        # y : 1D numerical ndarray of shape (m,)
        X = self._add_bias(X)

        if self.solver == 'gd':
            self._fit_gd(X, y)
        elif self.solver == 'normal':
            self._fit_normal(X, y)
        else:
            raise ValueError(f"Unknown solver '{self.solver}'. Use 'gd' or 'normal'.")

        return self

    def predict(self, X):
        # X must have the same number of features as training data
        if self.w_ is None:
            raise RuntimeError("LinearRegression is not fitted yet. Call fit() first.")
        return self._add_bias(X) @ self.w_