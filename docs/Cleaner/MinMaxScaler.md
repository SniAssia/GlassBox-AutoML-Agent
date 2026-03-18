# MinMaxScaler

**Module:** `cleaner.min_max_scaler`  
**Inherits:** —  
**Pattern:** `fit / transform / fit_transform`

---

## Overview

`MinMaxScaler` scales numerical features to a fixed range of $[0, 1]$ by shifting and rescaling based on the minimum and maximum values of each column. It preserves the shape of the original distribution and is useful when the algorithm requires bounded inputs or when preserving zero values in sparse data matters.

---

## Mathematical Formula

For each value $x$ in a column:

$$x_{\text{scaled}} = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

Where:
- $x_{\min}$ = minimum value of the column, computed on training data
- $x_{\max}$ = maximum value of the column, computed on training data

The minimum and maximum are computed from scratch using the `eda.statistics` module (no NumPy math functions).

---

## Design Assumptions

- `X` is assumed to be a **2D numerical NumPy ndarray** of shape `(n_samples, n_features)`.
- Encoding of categorical columns must be performed **upstream** before calling this scaler. By design, the GlassBox pipeline guarantees this ordering.

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| — | — | No constructor parameters. |

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `min_` | `ndarray` of shape `(n_features,)` | Per-column minimum learned during `fit()`. `None` before fitting. |
| `max_` | `ndarray` of shape `(n_features,)` | Per-column maximum learned during `fit()`. `None` before fitting. |

> The trailing underscore convention signals that these attributes are learned — they do not exist until `fit()` is called.

---

## Methods

### `fit(X)`

Learns the minimum and maximum of each column from the training data. Stores them in `min_` and `max_`. Does not modify `X`.

**Parameters:**
- `X` : 2D numerical ndarray of shape `(n_samples, n_features)`

**Returns:** `self`

---

### `transform(X)`

Applies MinMax scaling to `X` using the stored `min_` and `max_`. Can be called on training data, test data, or any new data — always uses the statistics learned from training.

> Note: test data values outside the $[x_{\min}, x_{\max}]$ range learned during training will produce scaled values outside $[0, 1]$. This is expected behaviour.

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
| Column with zero range ($x_{\max} = x_{\min}$, all values identical) | Range is replaced by 1 to avoid division by zero. The scaled column becomes all zeros. |

---

## Usage Example

```python
from cleaner.min_max_scaler import MinMaxScaler
import numpy as np

X_train = np.array([[10.0, 1.0],
                    [20.0, 2.0],
                    [30.0, 3.0]])

X_test = np.array([[15.0, 1.5]])

scaler = MinMaxScaler()

# fit on training data only
X_train_scaled = scaler.fit_transform(X_train)

# apply same learned statistics to test data
X_test_scaled = scaler.transform(X_test)

print(X_train_scaled)
# [[0.  0. ]
#  [0.5 0.5]
#  [1.  1. ]]

print(X_test_scaled)
# [[0.25 0.25]]

print(scaler.min_)  # [10.  1.]
print(scaler.max_)  # [30.  3.]
```

---

## StandardScaler vs MinMaxScaler

| | StandardScaler | MinMaxScaler |
|---|---|---|
| Output range | Unbounded — centered at 0 | Always $[0, 1]$ |
| Sensitive to outliers | Less sensitive | More sensitive — outliers compress the rest |
| Preserves zero | No | Yes (if $x_{\min} = 0$) |
| Use when | Features follow a roughly normal distribution | Features have a known bounded range |

---

## Pipeline Position

```
Raw data → SimpleImputer → LabelEncoder / OneHotEncoder → MinMaxScaler → Model
```

`MinMaxScaler` must always come **after** encoding steps since it operates on numerical arrays only.
