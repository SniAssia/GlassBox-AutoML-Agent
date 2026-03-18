# LabelEncoder

**Module:** `cleaner.label_encoder`  
**Inherits:** —  
**Pattern:** `fit / transform / fit_transform`

---

## Overview

`LabelEncoder` converts a **single categorical column** into integer codes. Each unique category is assigned a unique integer, making the column usable by numerical models.

It is designed for **ordinal** categorical data — data where the order of categories carries meaning (e.g. `low < medium < high`). For nominal data (no inherent order, e.g. `cat`, `dog`, `fish`), use `OneHotEncoder` instead.

---

## Encoding Example

| Original | Encoded |
|---|---|
| `'cat'` | `0` |
| `'dog'` | `1` |
| `'fish'` | `2` |

Categories are sorted **alphabetically** during `fit()` to ensure consistent and reproducible encoding.

---

## Design Assumptions

- Operates on a **single 1D NumPy ndarray** of categorical values.
- Input must contain no missing values — run `SimpleImputer` first.
- Categories are sorted alphabetically at fit time.

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
| `mapping_` | `dict` | Maps each category to its integer code. `None` before fitting. |

> The trailing underscore convention signals that these attributes are learned — they do not exist until `fit()` is called.

---

## Methods

### `fit(X)`

Learns the unique categories and assigns each an integer code in alphabetical order. Stores the mapping in `classes_` and `mapping_`.

**Parameters:**
- `X` : 1D ndarray of categorical values, shape `(n_samples,)`

**Returns:** `self`

---

### `transform(X)`

Replaces each category in `X` with its learned integer code. Unseen categories (not present during `fit()`) are assigned `-1`.

**Parameters:**
- `X` : 1D ndarray of categorical values, shape `(n_samples,)`

**Returns:** 1D integer ndarray of shape `(n_samples,)`

**Raises:** `RuntimeError` if called before `fit()`.

---

### `fit_transform(X)`

Convenience method — equivalent to calling `fit(X)` then `transform(X)` in one step.

> **Only use on training data.** Calling `fit_transform` on test data leaks label mappings and invalidates evaluation metrics.

**Parameters:**
- `X` : 1D ndarray of categorical values, shape `(n_samples,)`

**Returns:** 1D integer ndarray of shape `(n_samples,)`

---

## Unseen Categories

If `transform()` encounters a category not seen during `fit()`, it assigns the code `-1` (unknown). This is done using `dict.get(value, -1)` which returns the default `-1` when the key is not found.

```python
# example
encoder.fit(np.array(['cat', 'dog', 'fish']))
encoder.transform(np.array(['cat', 'bird']))
# → [0, -1]   ← 'bird' was never seen during fit
```

---

## Usage Example

```python
from cleaner.label_encoder import LabelEncoder
import numpy as np

X_train = np.array(['low', 'medium', 'high', 'medium', 'low'])
X_test  = np.array(['high', 'low', 'unknown_level'])

encoder = LabelEncoder()

X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded  = encoder.transform(X_test)

print(encoder.classes_)        # ['high' 'low' 'medium']
print(encoder.mapping_)        # {'high': 0, 'low': 1, 'medium': 2}
print(X_train_encoded)         # [1 2 0 2 1]
print(X_test_encoded)          # [ 0  1 -1]   ← 'unknown_level' → -1
```

---

## LabelEncoder vs OneHotEncoder

| | LabelEncoder | OneHotEncoder |
|---|---|---|
| Output shape | `(n_samples,)` — 1D | `(n_samples, n_categories + 1)` — 2D |
| Implies order | Yes — integers imply ranking | No — binary columns are order-free |
| Use for | Ordinal data (`low/medium/high`) | Nominal data (`cat/dog/fish`) |
| Risk | Model may incorrectly learn ordering from integer codes | No such risk |

---

## Pipeline Position

```
Raw data → SimpleImputer → LabelEncoder (ordinal cols) → Scaler → Model
```

`LabelEncoder` comes after `SimpleImputer` (no missing values allowed) and before scaling (output is numerical).
