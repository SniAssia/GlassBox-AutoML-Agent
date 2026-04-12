import numpy as np


def is_boolean(x):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("is_boolean requires at least one value")

    values = np.unique(x)
    return np.all(np.isin(values, [0, 1, True, False]))


def is_numerical(x):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("is_numerical requires at least one value")

    if is_boolean(x):
        return False

    try:
        np.asarray(x, dtype=float)
        return True
    except (ValueError, TypeError):
        return False


def is_categorical(x):
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError("is_categorical requires at least one value")

    return not is_boolean(x) and not is_numerical(x)


def infer_type(x):
    if is_boolean(x):
        return "boolean"
    if is_numerical(x):
        return "numerical"
    return "categorical"


def infer_types(X):
    X = np.asarray(X)

    if X.ndim != 2:
        raise ValueError("infer_types expects a 2D array")

    return [infer_type(X[:, i]) for i in range(X.shape[1])]