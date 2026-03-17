import numpy as np

from eda.statistics import mean, stdev


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("pearson expects two 1D arrays")
    if x.size != y.size:
        raise ValueError("x and y must have the same length")
    if x.size <= 1:
        raise ValueError("pearson requires at least two values per variable")

    mu_x = mean(x)
    mu_y = mean(y)

    sigma_x = stdev(x, mu_x)
    sigma_y = stdev(y, mu_y)

    if sigma_x == 0 or sigma_y == 0:
        return 0.0

    covariance = np.sum((x - mu_x) * (y - mu_y)) / (x.size - 1)
    return covariance / (sigma_x * sigma_y)


def pearson_correlation_matrix(X):
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError("pearson_correlation_matrix expects a 2D array")
    if X.shape[0] <= 1:
        raise ValueError("pearson_correlation_matrix requires at least two rows")

    n_features = X.shape[1]
    corr = np.eye(n_features, dtype=float) # Identity matrix In

    for i in range(n_features):
        for j in range(i + 1, n_features):
            value = pearson(X[:, i], X[:, j])
            corr[i, j] = value
            corr[j, i] = value

    return corr