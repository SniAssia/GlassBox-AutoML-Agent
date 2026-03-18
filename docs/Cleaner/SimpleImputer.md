# SimpleImputer

**Module:** `cleaner.simple_imputer`  
**Inherits:** —  
**Pattern:** `fit / transform / fit_transform`

---

## Overview

`SimpleImputer` fills missing values in a dataset using a simple statistical strategy. It handles both **numerical** and **categorical** columns automatically by detecting the column type at fit time.

Missing values are detected as either `np.nan` or `None`.

---

## Strategy

| Column type | Fill strategy |
|---|---|
| Numerical | Mean or Median — chosen via `method` parameter |
| Categorical | Mode (most frequent value) — always, regardless of `method` |

---

## Design Assumptions

- `X` is a **2D NumPy ndarray** that may contain a mix of numerical and categorical columns, as well as `np.nan` or `None` values.
- Column type is **auto-detected** at fit time: a column is considered numerical if all its non-missing values can be cast to `float`. Otherwise it is treated as categorical.
- Mean, median, and mode are computed from scratch using the `eda.statistics` module.

---

## Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| — | — | — | No constructor parameters. `method` is passed at `fit()` time. |

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `fill_values_` | `list` of length `n_features` | One fill value per column learned during `fit()`. `None` before fitting. |

> The trailing underscore convention signals that this attribute is learned — it does not exist until `fit()` is called.

---

## Methods

### `fit(X, method='mean')`

Learns one fill value per column from the training data. For numerical columns, computes mean or median depending on `method`. For categorical columns, always computes mode. Does not modify `X`.

**Parameters:**
- `X` : 2D ndarray of shape `(n_samples, n_features)`, may contain `np.nan` or `None`
- `method` : `str`, either `'mean'` or `'median'`. Only applies to numerical columns. Default is `'mean'`.

**Returns:** `self`

---

### `transform(X)`

Replaces all missing values in `X` using the fill values learned during `fit()`. Operates on a **copy** of `X` — the original array is never modified.

**Parameters:**
- `X` : 2D ndarray of shape `(n_samples, n_features)`, may contain `np.nan` or `None`

**Returns:** Filled ndarray of shape `(n_samples, n_features)` with no missing values.

**Raises:** `RuntimeError` if called before `fit()`.

---

### `fit_transform(X, method='mean')`

Convenience method — equivalent to calling `fit(X, method)` then `transform(X)` in one step.

> **Only use on training data.** Calling `fit_transform` on test data leaks test statistics and invalidates evaluation metrics.

**Parameters:**
- `X` : 2D ndarray of shape `(n_samples, n_features)`
- `method` : `str`, either `'mean'` or `'median'`. Default is `'mean'`.

**Returns:** Filled ndarray of shape `(n_samples, n_features)`

---

## Internal Helpers

### `_is_missing(col)`

Returns a boolean mask of shape `(n_samples,)` — `True` where the value is `np.nan` or `None`, `False` elsewhere.

### `_is_numerical(col)`

Returns `True` if all non-missing values in the column can be cast to `float`. Returns `False` if any value raises a `ValueError` or `TypeError` during casting — indicating a categorical column.

---

## Why Copy X in transform?

NumPy arrays are passed by reference. Modifying `X` directly inside `transform()` would corrupt the caller's original data silently. Using `X.copy()` ensures the original array is always preserved.

---

## Why Store fill_values_ Instead of Recomputing in transform?

The fill values must be learned **only from training data**. If they were recomputed inside `transform()`, applying it to test data would leak test statistics — a form of data leakage that makes evaluation metrics unreliable. By storing them during `fit()`, the same values are applied consistently to any dataset.

---

## Usage Example

```python
from cleaner.simple_imputer import SimpleImputer
import numpy as np

X_train = np.array([[1.0,   'cat', 3.0],
                    [None,  'dog', None],
                    [3.0,   None,  5.0],
                    [4.0,   'cat', 2.0]], dtype=object)

imputer = SimpleImputer()

# fit on training data — numerical columns use mean, categorical use mode
X_train_filled = imputer.fit_transform(X_train, method='mean')

print(X_train_filled)
# [[1.0,   'cat', 3.0 ]
#  [2.67,  'dog', 3.33]   ← numerical filled with mean, categorical with mode
#  [3.0,   'cat', 5.0 ]
#  [4.0,   'cat', 2.0 ]]

print(imputer.fill_values_)
# [2.666..., 'cat', 3.333...]
```

---

## Pipeline Position

```
Raw data → SimpleImputer → LabelEncoder / OneHotEncoder → Scaler → Model
```

`SimpleImputer` must always come **first** in the pipeline — encoders and scalers cannot handle missing values.
