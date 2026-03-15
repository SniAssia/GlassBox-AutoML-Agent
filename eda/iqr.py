import numpy as np

from eda.statistics import median

FACTOR = 1.5

def quartiles(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size

    if n == 0:
        raise ValueError("quartiles requires at least one value")
    if n == 1:
        return x[0], x[0], x[0]

    q2 = median(x)
    mid = n // 2

    if n % 2 == 0:
        lower = x[:mid]
        upper = x[mid:]
    else:
        lower = x[:mid]
        upper = x[mid + 1:]

    q1 = median(lower)
    q3 = median(upper)
    return q1, q2, q3


def iqr(x):
    q1, _, q3 = quartiles(x)
    return q3 - q1


def iqr_bounds(x, factor=FACTOR):
    q1, _, q3 = quartiles(x)
    spread = q3 - q1
    lower = q1 - factor * spread
    upper = q3 + factor * spread
    return lower, upper


def iqr_outliers(x, factor=FACTOR):
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("iqr_outliers requires at least one value")

    lower, upper = iqr_bounds(x, factor)
    # check if outlier or not
    return (x < lower) | (x > upper)


def cap_outliers_iqr(x, factor=FACTOR):
    x = np.asarray(x, dtype=float)

    if x.size == 0:
        raise ValueError("cap_outliers_iqr requires at least one value")

    lower, upper = iqr_bounds(x, factor)
    # any values lower than `lower` become equal to lower, same for upper, that's clipping
    return np.clip(x, lower, upper)