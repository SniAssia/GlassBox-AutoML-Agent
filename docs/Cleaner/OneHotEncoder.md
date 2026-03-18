# OneHotEncoder

**Module:** `cleaner.one_hot_encoder`  
**Inherits:** —  
**Pattern:** `fit / transform / fit_transform`

---

## Overview

`OneHotEncoder` converts a **single categorical column** into a **binary matrix**. Each unique category gets its own column — a `1` is placed in the column corresponding to the category of the sample, and `0` everywhere else. An additional **unknown column** is appended to handle categories not seen during training.

It is designed for **nominal** categorical data — data where categories have no inherent order (e.g. `cat`, `dog`, `fish`). For ordinal data (e.g. `low < medium < high`), use `LabelEncoder` instead.

---

## Encoding Example

Training categories: `['cat', 'dog', 'fish']`

| Original | cat | dog | fish | unknown |
|---|---|---|---|---|
| `'cat'` | 1 | 0 | 0 | 0 |
| `'dog'` | 0 | 1 | 0 | 0 |
| `'fish'` | 0 | 0 | 1 | 0 |
| `'bird'` | 0 | 0 | 0 | 1 |  ← unseen category

The output shape is always `(n_samples, n_categories + 1)` — the `+1` is for the unknown column.

---

## Design Assumptions

- Operates on a **single 1D NumPy ndarray** of categorical values.
- Input must contain no missing values — run `SimpleImputer` first.
- Categories are sorted **alphabetically** during `fit()` to ensure consistent and reproducible column ordering.

---

## Parameters

| Parameter | Type | Description |
|---|---|---|
| — | — | No constructor parameters. |

---

## Attributes

| Attribute | Type | Description |
|---|---|---|
| `classes_` | `ndarray` of shape `(n_categories,)` | Unique categories learned during `fit()`, sorted alphabetically. `None` before fitting. |
| `mapping_` | `dict` | Maps each category to its column index in the output matrix. `None` before fitting. |

> The trailing underscore convention signals that these attributes are learned — they do not exist until `fit()` is called.

---

## Methods

### `fit(X)`

Learns the unique categories and assigns each a column index in alphabetical order. Stores the mapping in `classes_` and `mapping_`.

**Parameters:**
- `X` : 1D ndarray of categorical values, shape `(n_samples,)`

**Returns:** `self`

---

### `transform(X)`

Returns a binary matrix of shape `(n_samples, n_categories + 1)`. For each sample, places a `1` in the column corresponding to its category. If the category was not seen during `fit()`, places a `1` in the last column (the unknown column).

**Parameters:**
- `X` : 1D ndarray of categorical values, shape `(n_samples,)`

**Returns:** 2D integer ndarray of shape `(n_samples, n_categories + 1)`

**Raises:** `RuntimeError` if called before `fit()`.

---

### `fit_transform(X)`

Convenience method — equivalent to calling `fit(X)` then `transform(X)` in one step.

> **Only use on training data.** Calling `fit_transform` on test data leaks category mappings and invalidates evaluation metrics.

**Parameters:**
- `X` : 1D ndarray of categorical values, shape `(n_samples,)`

**Returns:** 2D integer ndarray of shape `(n_samples, n_categories + 1)`

---

## Unseen Categories

If `transform()` encounters a category not seen during `fit()`, it fires the **last column** (the unknown column) instead of raising an error. This makes the encoder robust to distribution shifts between training and test data.

```python
# example
encoder.fit(np.array(['cat', 'dog', 'fish']))
# classes_ = ['cat', 'dog', 'fish']
# output columns = [cat, dog, fish, unknown]

encoder.transform(np.array(['dog', 'bird']))
# → [[0, 1, 0, 0],   ← 'dog' known
#    [0, 0, 0, 1]]   ← 'bird' unseen → unknown column fires
```

---

## Usage Example

```python
from cleaner.one_hot_encoder import OneHotEncoder
import numpy as np

X_train = np.array(['cat', 'dog', 'fish', 'cat', 'dog'])
X_test  = np.array(['fish', 'cat', 'bird'])

encoder = OneHotEncoder()

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded  = encoder.transform(X_test)

print(encoder.classes_)
# ['cat' 'dog' 'fish']

print(encoder.mapping_)
# {'cat': 0, 'dog': 1, 'fish': 2}

print(X_train_encoded)
# [[1 0 0 0]   cat
#  [0 1 0 0]   dog
#  [0 0 1 0]   fish
#  [1 0 0 0]   cat
#  [0 1 0 0]]  dog

print(X_test_encoded)
# [[0 0 1 0]   fish
#  [1 0 0 0]   cat
#  [0 0 0 1]]  bird → unknown column
```

---

## Why Not Use LabelEncoder for Nominal Data?

`LabelEncoder` assigns integers like `cat=0, dog=1, fish=2`. This implies a numerical ordering — the model might incorrectly learn that `fish > dog > cat`. `OneHotEncoder` avoids this by giving each category its own independent binary column, making no assumptions about order.

---

## LabelEncoder vs OneHotEncoder

| | LabelEncoder | OneHotEncoder |
|---|---|---|
| Output shape | `(n_samples,)` — 1D | `(n_samples, n_categories + 1)` — 2D |
| Implies order | Yes — integers imply ranking | No — binary columns are order-free |
| Use for | Ordinal data (`low/medium/high`) | Nominal data (`cat/dog/fish`) |
| Unseen category | Assigned `-1` | Unknown column fires |
| Dimensionality | No increase | Adds one column per category |

---

## Pipeline Position

```
Raw data → SimpleImputer → OneHotEncoder (nominal cols) → Scaler → Model
```

`OneHotEncoder` comes after `SimpleImputer` (no missing values allowed) and before scaling. Note that the output is already binary `(0/1)` — applying `MinMaxScaler` or `StandardScaler` to one-hot columns is generally unnecessary and sometimes harmful.
