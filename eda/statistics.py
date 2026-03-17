import numpy as np

def mean(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("mean requires at least one value")
    return np.sum(x) / x.size

def median(x):
    x = np.sort(np.asarray(x))
    n = x.size
    if n == 0:
        raise ValueError("median requires at least one value")
    mid = n // 2
    if n % 2 == 1:
        return x[mid]
    return (x[mid - 1] + x[mid]) / 2

def mode(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0: 
        raise ValueError("mode requires at least one value")
    values, counts = np.unique(x, return_counts=True)
    return values[counts == np.max(counts)][0]

# sample stdev
def stdev(x, mu=None):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= 1: 
        raise ValueError("standard deviation needs at least two values")
    
    if mu is None: diff = x - mean(x)
    else: diff = x - mu
    return np.sqrt(np.sum(diff * diff) / (n - 1))

# adjusted Fisher-Pearson skewness
def skewness(x, mu=None, sigma=None):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= 2:
        raise ValueError("skewness requires at least three values")

    if mu is None:
        mu = mean(x)
    if sigma is None:
        sigma = stdev(x, mu)  # sample std

    if sigma == 0:
        return 0.0

    z = (x - mu) / sigma
    return np.sum(z ** 3) * (n / ((n - 1) * (n - 2)))

# Sample excess kurtosis
def kurtosis(x, mu=None, sigma=None):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n <= 3:
        raise ValueError("kurtosis requires at least four values")

    if mu is None:
        mu = mean(x)
    if sigma is None:
        sigma = stdev(x, mu)  # sample std

    if sigma == 0:
        return 0.0
    
    z = (x - mu) / sigma
    c1 = n * (n + 1) / ( (n - 1) * (n - 2) * (n - 3) )
    c2 = 3 * (n - 1) ** 2 / ( (n - 2) * (n - 3) )
    return c1 * np.sum(z ** 4) - c2



# Soufiane comment: you forgot min and max. I added them because i need them in minMaxScaler

def min_val(x):
    # x is a 1D numerical ndarray
    minimum = x[0]
    for val in x[1:]:
        if val < minimum:
            minimum = val
    return minimum


def max_val(x):
    # x is a 1D numerical ndarray
    maximum = x[0]
    for val in x[1:]:
        if val > maximum:
            maximum = val
    return maximum