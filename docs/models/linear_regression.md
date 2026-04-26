# Linear Regression

## Overview

`LinearRegression` predicts a continuous target using a linear function of the input features. The bias term is absorbed into the weight vector by prepending a column of ones to the design matrix, so prediction is written as:

$$f(X) = Xw$$

The implementation supports two solvers:

- `solver='gd'` for batch gradient descent
- `solver='normal'` for a closed-form least-squares solution

Implementation source: [linear_regression.py](c:/Users/dynabook/OneDrive/Bureau/CC_CI1/S6/Artificial_intelligence/project/GlassBox-AutoML-Agent/models/linear_regression.py)

## Cost Function

The model minimizes Mean Squared Error:

$$C(w) = \frac{1}{2m}\|Xw - y\|^2$$

Its gradient is:

$$\nabla_w C = \frac{1}{m} X^T(Xw - y)$$

Because this objective is convex, linear regression has a global minimum.

## Solver Behavior

### Gradient Descent

The `gd` solver:

- initializes weights at zero
- performs batch updates using the gradient
- records the cost history in `costs_`
- supports early stopping through `tol`

Update rule:

$$w := w - \alpha \frac{1}{m}X^T(Xw - y)$$

### Closed-Form Solver

The classical normal-equation view is:

$$w = (X^TX)^{-1}X^Ty$$

However, the current GlassBox implementation is intentionally more robust than a strict inverse-based formula. The `normal` solver uses the Moore-Penrose pseudoinverse:

$$w = X^{+}y$$

In code, this is implemented with:

```python
np.linalg.pinv(X) @ y
```

That means the closed-form path:

- still produces a least-squares solution
- avoids the common singular-matrix crash of inverse-only implementations
- handles redundant or highly correlated features more gracefully

## Singular Matrix Note

Older inverse-based implementations often fail with:

```python
numpy.linalg.LinAlgError: Singular matrix
```

This project avoids that specific failure in the `normal` solver by using the pseudoinverse instead of `np.linalg.inv(X.T @ X)`.

## Why Keep Gradient Descent?

Even with the pseudoinverse improvement, gradient descent is still valuable because it:

- scales better to larger feature spaces
- supports early stopping
- gives cost traces for debugging
- fits naturally into iterative AutoML workflows

So the closed-form solver is convenient, but gradient descent remains a strong default for general-purpose use.

## Usage

### Gradient Descent

```python
from models.linear_regression import LinearRegression
import numpy as np

X_train = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
])

y_train = np.array([5.0, 8.0, 11.0, 14.0])

model = LinearRegression(solver="gd", lr=0.01, n_epochs=1000, tol=1e-6)
model.fit(X_train, y_train)

X_test = np.array([[5.0, 6.0]])
predictions = model.predict(X_test)

print(predictions)
print(model.w_)
print(model.costs_[-1])
```

### Closed-Form Solver

```python
from models.linear_regression import LinearRegression
import numpy as np

X_train = np.array([
    [1.0, 2.0],
    [2.0, 3.0],
    [3.0, 4.0],
    [4.0, 5.0],
])

y_train = np.array([5.0, 8.0, 11.0, 14.0])

model = LinearRegression(solver="normal")
model.fit(X_train, y_train)

X_test = np.array([[5.0, 6.0]])
predictions = model.predict(X_test)

print(predictions)
print(model.w_)
# model.costs_ stays empty because there is no iterative optimization
```

## Choosing Between Solvers

```python
# small dataset, few features
model = LinearRegression(solver="normal")

# larger or more variable workloads
model = LinearRegression(solver="gd", lr=0.01, n_epochs=2000)
```

## Pipeline Example

```python
from cleaner.simple_imputer import SimpleImputer
from cleaner.standard_Scaler import StandardScaler
from models.linear_regression import LinearRegression
import numpy as np

X_raw = np.array([
    [1.0, np.nan],
    [2.0, 3.0],
    [np.nan, 4.0],
    [4.0, 5.0],
], dtype=float)

y = np.array([3.0, 5.0, 7.0, 9.0])

imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X_raw)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

model = LinearRegression(solver="gd", lr=0.1, n_epochs=1000)
model.fit(X_scaled, y)

X_new = np.array([[3.0, 6.0]])
X_new_imputed = imputer.transform(X_new)
X_new_scaled = scaler.transform(X_new_imputed)
print(model.predict(X_new_scaled))
```

Use `transform()` on new data, not `fit_transform()`, so preprocessing stays tied to training-time statistics.
