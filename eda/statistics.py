import numpy as np

class EDA:
    def __init__(self):
        pass

    def mean(self, x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            raise ValueError("mean requires at least one value")
        return np.sum(x) / x.size

    def median(self, x):
        x = np.sort(np.asarray(x))
        n = x.size
        if n == 0:
            raise ValueError("median requires at least one value")
        mid = n // 2
        if n % 2 == 1:
            return x[mid]
        return (x[mid - 1] + x[mid]) / 2

    def mode(self, x):
        x = np.asarray(x)
        if x.size == 0: 
            raise ValueError("mode requires at least one value")
        values, counts = np.unique(x, return_counts=True)
        return values[counts == np.max(counts)][0]

    def stdev(self, x, mu=None):
        x = np.asarray(x, dtype=float)
        n = x.size
        if n <= 1: 
            raise ValueError("standard deviation needs at least two values")
        if mu is None: mu = self.mean(x)
        else: diff = x - mu
        diff = x - (mu if mu is not None else self.mean(x))
        return np.sqrt(np.sum(diff * diff) / (n - 1))

    def skewness(self, x, mu=None, sigma=None):
        x = np.asarray(x, dtype=float)
        n = x.size
        if n <= 2: raise ValueError("skewness requires at least three values")
        if mu is None: mu = self.mean(x)
        if sigma is None: sigma = self.stdev(x, mu)
        if sigma == 0: return 0.0
        z = (x - mu) / sigma
        return np.sum(z ** 3) * (n / ((n - 1) * (n - 2)))

    def kurtosis(self, x, mu=None, sigma=None):
        x = np.asarray(x, dtype=float)
        n = x.size
        if n <= 3: raise ValueError("kurtosis requires at least four values")
        if mu is None: mu = self.mean(x)
        if sigma is None: sigma = self.stdev(x, mu)
        if sigma == 0: return 0.0
        z = (x - mu) / sigma
        c1 = n * (n + 1) / ( (n - 1) * (n - 2) * (n - 3) )
        c2 = 3 * (n - 1) ** 2 / ( (n - 2) * (n - 3) )
        return c1 * np.sum(z ** 4) - c2

    def min_val(self, x):
        x = np.asarray(x, dtype=float)
        return np.min(x) if x.size > 0 else 0

    def max_val(self, x):
        x = np.asarray(x, dtype=float)
        return np.max(x) if x.size > 0 else 0

    def analyze(self, data, col_types):
        """
        Méthode appelée par l'Orchestrateur pour générer les stats du rapport.
        """
        report = {}
        for i, t in enumerate(col_types):
            col_data = data[:, i]
            col_info = {"type": t}
            
            if t == "numerical":
                # On essaie de convertir en float pour les calculs
                try:
                    numeric_data = col_data.astype(float)
                    col_info["mean"] = self.mean(numeric_data)
                    col_info["min"] = self.min_val(numeric_data)
                    col_info["max"] = self.max_val(numeric_data)
                except:
                    col_info["error"] = "Could not calculate stats"
            
            report[f"column_{i}"] = col_info
        return report