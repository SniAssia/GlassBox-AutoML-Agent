# StandardScaler

**Module:** `cleaner.standard_scaler`  
**Inherits:** —  
**Pattern:** `fit / transform / fit_transform`

---

## Overview

`StandardScaler` standardizes numerical features by removing the mean and scaling to unit variance. This transformation is also known as **Z-score normalization**. After scaling, each feature has a mean of 0 and a standard deviation of 1.

This is one of the most commonly used preprocessing steps before training machine learning models, especially those sensitive to feature magnitude (e.g. gradient descent based models like Linear and Logistic Regression).

---

## Mathematical Formula

For each value $x$ in a column:

$$x_{\text{scaled}} = \frac{x - \mu}{\sigma}$$

Where:
- $\mu$ = mean of the column, computed on training data
- $\sigma$ = standard deviation of the column, computed on training data

The mean and standard deviation are computed from scratch using the `eda.statistics` module (no NumPy math functions):

$$\mu = \frac{1}{n}\sum_{i=1}^{n} x_i \qquad \sigma = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}$$

---

## Design Assumptions

- `X` is assumed to be a **2D numerical NumPy ndarray** of shape `(n_samples, n_features)`.
- Encoding of categorical columns must be performed **upstream** before calling this scaler. By design, the GlassBox pipeline guarantees this ordering.
- Uses **population standard deviation** (divide by $n$, not $n-1$).

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| — | — | No constructor parameters. |

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `mean_` | `ndarray` of shape `(n_features,)` | Per-column mean learned during `fit()`. `None` before fitting. |
| `stadev_` | `ndarray` of shape `(n_features,)` | Per-column standard deviation learned during `fit()`. `None` before fitting. |

> The trailing underscore convention signals that these attributes are learned — they do not exist until `fit()` is called.

---

## Methods

### `fit(X)`

Learns the mean and standard deviation of each column from the training data. Stores them in `mean_` and `stadev_`. Does not modify `X`.

**Parameters:**
- `X` : 2D numerical ndarray of shape `(n_samples, n_features)`

**Returns:** `self`

---

### `transform(X)`

Applies Z-score normalization to `X` using the stored `mean_` and `stadev_`. Can be called on training data, test data, or any new data — always uses the statistics learned from training.

**Parameters:**
- `X` : 2D numerical ndarray of shape `(n_samples, n_features)`

**Returns:** Transformed ndarray of shape `(n_samples, n_features)`

**Raises:** `RuntimeError` if called before `fit()`.

---

### `fit_transform(X)`

Convenience method — equivalent to calling `fit(X)` then `transform(X)` in one step.

> **Only use on training data.** Calling `fit_transform` on test data leaks test statistics into the scaler and invalidates evaluation metrics.

**Parameters:**
- `X` : 2D numerical ndarray of shape `(n_samples, n_features)`

**Returns:** Transformed ndarray of shape `(n_samples, n_features)`

---

## Edge Cases

| Situation | Behaviour |
|---|---|
| Column with zero variance (all values identical, $\sigma = 0$) | Standard deviation is replaced by 1 to avoid division by zero. The scaled column becomes all zeros. |

---

## Usage Example

```python
from cleaner.standard_scaler import StandardScaler
import numpy as np

X_train = np.array([[1.0, 200.0],
                    [2.0, 400.0],
                    [3.0, 600.0]])

X_test = np.array([[4.0, 800.0]])

scaler = StandardScaler()

# fit on training data only
X_train_scaled = scaler.fit_transform(X_train)

# apply same learned statistics to test data
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
# Each column now has mean ≈ 0 and std ≈ 1

print(scaler.mean_)    # [2.  400.]
print(scaler.stadev_)  # [0.816... 163.2...]
```

---

## Pipeline Position

```
Raw data → SimpleImputer → LabelEncoder / OneHotEncoder → StandardScaler → Model
```

`StandardScaler` must always come **after** encoding steps since it operates on numerical arrays only.
