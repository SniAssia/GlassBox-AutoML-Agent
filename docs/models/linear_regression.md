# Linear Regression — Theory Notes

---

## 1. The Model

For multilinear regression with $n$ features, the prediction is:

$$f(\mathbf{x}_i) = w_1x_{i1} + w_2x_{i2} + \dots + w_nx_{in} + b = \mathbf{w}^T\mathbf{x}_i + b$$

To simplify, we absorb the bias $b$ into $\mathbf{w}$ by prepending a column of ones to $X$:

$$\mathbf{w} = \begin{bmatrix} b \\ w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}, \quad X = \begin{bmatrix} 1 & x_{11} & \cdots & x_{1n} \\ 1 & x_{21} & \cdots & x_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ 1 & x_{m1} & \cdots & x_{mn} \end{bmatrix}$$

So the prediction becomes simply:

$$f(X) = X\mathbf{w}$$

---

## 2. The Cost Function (MSE)

We use Mean Squared Error as the loss function:

$$C(\mathbf{w}) = \frac{1}{2m} \sum_{i=1}^{m} (f(\mathbf{x}_i) - y_i)^2$$

In matrix form:

$$C(\mathbf{w}) = \frac{1}{2m} \|X\mathbf{w} - \mathbf{y}\|^2 = \frac{1}{2m}(X\mathbf{w} - \mathbf{y})^T(X\mathbf{w} - \mathbf{y})$$

Expanding:

$$C(\mathbf{w}) = \frac{1}{2m}\left(\mathbf{w}^TX^TX\mathbf{w} - 2\mathbf{y}^TX\mathbf{w} + \mathbf{y}^T\mathbf{y}\right)$$

> The $\frac{1}{2}$ is a convenience factor — it cancels the $2$ that appears when differentiating, making the gradient formula cleaner.

---

## 3. Proof of Convexity

A function is convex if its **Hessian matrix is positive semi-definite** (all eigenvalues $\geq 0$).

---

### Step 1 — Compute the gradient of $C(\mathbf{w})$

Recall the expanded cost function:

$$C(\mathbf{w}) = \frac{1}{2m}\left(\mathbf{w}^TX^TX\mathbf{w} - 2\mathbf{y}^TX\mathbf{w} + \mathbf{y}^T\mathbf{y}\right)$$

We differentiate term by term using the matrix calculus rules from Section 5:

**Term 1:** $\mathbf{w}^TX^TX\mathbf{w}$

This is a quadratic form $\mathbf{w}^TA\mathbf{w}$ with $A = X^TX$.
Since $X^TX$ is always symmetric ($A = A^T$), Rule 1 gives:

$$\nabla_\mathbf{w}(\mathbf{w}^TX^TX\mathbf{w}) = 2X^TX\mathbf{w}$$

**Term 2:** $-2\mathbf{y}^TX\mathbf{w}$

This is a linear form $\mathbf{a}^T\mathbf{w}$ with $\mathbf{a} = -2X^T\mathbf{y}$.
Rule 2 gives:

$$\nabla_\mathbf{w}(-2\mathbf{y}^TX\mathbf{w}) = -2X^T\mathbf{y}$$

**Term 3:** $\mathbf{y}^T\mathbf{y}$

This is a constant with respect to $\mathbf{w}$.
Rule 3 gives:

$$\nabla_\mathbf{w}(\mathbf{y}^T\mathbf{y}) = 0$$

**Summing all three terms** and factoring out $\frac{1}{2m}$:

$$\nabla_\mathbf{w} C = \frac{1}{2m}\left(2X^TX\mathbf{w} - 2X^T\mathbf{y} + 0\right)$$

The $2$s cancel with the $\frac{1}{2}$:

$$\boxed{\nabla_\mathbf{w} C = \frac{1}{m}(X^TX\mathbf{w} - X^T\mathbf{y})}$$

> This is exactly why the $\frac{1}{2}$ was placed in the cost function — it was designed to cancel the $2$ from differentiation and keep the gradient clean.

---

### Step 2 — Compute the Hessian of $C(\mathbf{w})$

The Hessian is the derivative of the gradient with respect to $\mathbf{w}$:

$$H = \nabla^2_\mathbf{w} C = \nabla_\mathbf{w} \left[\frac{1}{m}(X^TX\mathbf{w} - X^T\mathbf{y})\right]$$

We differentiate term by term:

**Term 1:** $\frac{1}{m}X^TX\mathbf{w}$

This is a linear function of $\mathbf{w}$ of the form $A\mathbf{w}$ with $A = \frac{1}{m}X^TX$.
Its derivative with respect to $\mathbf{w}$ is simply the matrix $A$:

$$\nabla_\mathbf{w}\left(\frac{1}{m}X^TX\mathbf{w}\right) = \frac{1}{m}X^TX$$

**Term 2:** $\frac{1}{m}X^T\mathbf{y}$

This does not depend on $\mathbf{w}$ — it is a constant vector.
Its derivative is zero:

$$\nabla_\mathbf{w}\left(\frac{1}{m}X^T\mathbf{y}\right) = 0$$

**Therefore the Hessian is:**

$$\boxed{H = \frac{1}{m}X^TX}$$

---

### Step 3 — Show $H$ is positive semi-definite

A matrix $H$ is positive semi-definite if for **any** non-zero vector $\mathbf{v}$:

$$\mathbf{v}^T H \mathbf{v} \geq 0$$

Substituting our Hessian:

$$\mathbf{v}^T H \mathbf{v} = \mathbf{v}^T \left(\frac{1}{m}X^TX\right) \mathbf{v} = \frac{1}{m} \mathbf{v}^T X^T X \mathbf{v}$$

Let $\mathbf{u} = X\mathbf{v}$. Then $\mathbf{v}^TX^TX\mathbf{v} = \mathbf{u}^T\mathbf{u} = \|\mathbf{u}\|^2$:

$$\mathbf{v}^T H \mathbf{v} = \frac{1}{m}\|X\mathbf{v}\|^2 \geq 0$$

Since a squared norm is always $\geq 0$, the Hessian is **positive semi-definite**. Therefore $C(\mathbf{w})$ is **convex**. ✅

This means there is exactly **one global minimum** — no local minima to get stuck in.

---

## 4. The Normal Equation (Closed Form)

Since $C$ is convex, the global minimum is where the gradient equals zero:

$$\nabla_\mathbf{w} C = 0$$

$$\frac{1}{m}(X^TX\mathbf{w} - X^T\mathbf{y}) = 0$$

$$X^TX\mathbf{w} = X^T\mathbf{y}$$

These are called the **normal equations**. If $X^TX$ is invertible, we multiply both sides by $(X^TX)^{-1}$:

$$\boxed{\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}}$$

This gives the exact optimal weights **in one shot**, with no iterations. ✅

---

## 5. Matrix Calculus Rules Used

Three rules were used to derive the gradient from $C(\mathbf{w})$.

---

### Rule 1 — Quadratic form

$$\nabla_\mathbf{w}(\mathbf{w}^T A \mathbf{w}) = (A + A^T)\mathbf{w}$$

**Why?** The $i$-th component of the gradient is $\frac{\partial}{\partial w_i}(\mathbf{w}^TA\mathbf{w})$. Expanding the quadratic form:

$$\mathbf{w}^TA\mathbf{w} = \sum_j \sum_k A_{jk} w_j w_k$$

Differentiating with respect to $w_i$:

$$\frac{\partial}{\partial w_i}\sum_j \sum_k A_{jk} w_j w_k = \sum_k A_{ik} w_k + \sum_j A_{ji} w_j = (A\mathbf{w})_i + (A^T\mathbf{w})_i$$

Stacking all components gives $(A + A^T)\mathbf{w}$.

**Special case — symmetric $A$ (where $A^T = A$):**

$$(A + A^T)\mathbf{w} = 2A\mathbf{w}$$

**Why is $X^TX$ always symmetric?**

$$(X^TX)^T = X^T(X^T)^T = X^TX \quad \checkmark$$

So applying Rule 1 to $\mathbf{w}^TX^TX\mathbf{w}$:

$$\nabla_\mathbf{w}(\mathbf{w}^TX^TX\mathbf{w}) = 2X^TX\mathbf{w}$$

---

### Rule 2 — Linear form

$$\nabla_\mathbf{w}(\mathbf{a}^T\mathbf{w}) = \mathbf{a}$$

**Why?** $\mathbf{a}^T\mathbf{w} = \sum_i a_i w_i$. Differentiating with respect to $w_i$ gives simply $a_i$. Stacking all components gives $\mathbf{a}$.

**Applied to our term** $-2\mathbf{y}^TX\mathbf{w}$:

First rewrite it as $\mathbf{a}^T\mathbf{w}$ where $\mathbf{a} = -2X^T\mathbf{y}$ (since $\mathbf{y}^TX\mathbf{w} = (X^T\mathbf{y})^T\mathbf{w}$):

$$\nabla_\mathbf{w}(-2\mathbf{y}^TX\mathbf{w}) = -2X^T\mathbf{y}$$

---

### Rule 3 — Constant

$$\nabla_\mathbf{w}(\mathbf{y}^T\mathbf{y}) = 0$$

**Why?** $\mathbf{y}^T\mathbf{y} = \sum_i y_i^2$ does not contain $\mathbf{w}$ at all. Its derivative with respect to any $w_i$ is zero.

---

## 6. Why Gradient Descent Instead of the Normal Equation?

Although the normal equation gives the exact solution in one shot, it requires computing $(X^TX)^{-1}$ — a **matrix inversion** of complexity $O(n^3)$, where $n$ is the number of features. For small datasets this is fine, but it becomes extremely slow or numerically unstable for large ones. Additionally, if $X^TX$ is not invertible (which happens when features are correlated or redundant), the normal equation breaks entirely. Gradient descent, by contrast, never inverts a matrix, scales well to millions of features, and handles ill-conditioned data gracefully. Since GlassBox is a general-purpose AutoML library that cannot know in advance how large or well-conditioned the input will be, gradient descent is the right default choice.

---

## 7. Gradient Descent Update Rule

At each iteration, weights are updated by moving in the direction of the negative gradient:

$$\mathbf{w} := \mathbf{w} - \alpha \nabla_\mathbf{w} C = \mathbf{w} - \frac{\alpha}{m} X^T(X\mathbf{w} - \mathbf{y})$$

Where $\alpha$ is the **learning rate** — controls the step size at each iteration.

---

## 8. Epoch vs Iteration

| Term | Meaning |
|---|---|
| **Iteration** | One weight update using one sample or one batch |
| **Epoch** | One full pass through the entire training dataset |

In **Batch Gradient Descent** (our implementation), one epoch = one iteration, because we use all $m$ samples to compute the gradient before each update.

---

## 9. Usage Examples

### Basic usage — Gradient Descent

```python
from models.linear_regression import LinearRegression
import numpy as np

X_train = np.array([[1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [4.0, 5.0]])

y_train = np.array([5.0, 8.0, 11.0, 14.0])

model = LinearRegression(solver='gd', lr=0.01, n_epochs=1000)
model.fit(X_train, y_train)

X_test = np.array([[5.0, 6.0]])
predictions = model.predict(X_test)

print(predictions)       # → [17.0] (approximately)
print(model.w_)          # → [bias, w1, w2]
print(model.costs_[-1])  # → final cost after training
```

---

### Basic usage — Normal Equation

```python
from models.linear_regression import LinearRegression
import numpy as np

X_train = np.array([[1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [4.0, 5.0]])

y_train = np.array([5.0, 8.0, 11.0, 14.0])

model = LinearRegression(solver='normal')
model.fit(X_train, y_train)

X_test = np.array([[5.0, 6.0]])
predictions = model.predict(X_test)

print(predictions)   # → [17.0] (exact)
print(model.w_)      # → [bias, w1, w2]
# model.costs_ is empty — no iterations with the normal equation
```

---

### Choosing between solvers

```python
# small dataset, few features → normal equation is faster and exact
model = LinearRegression(solver='normal')

# large dataset, many features → gradient descent scales better
model = LinearRegression(solver='gd', lr=0.01, n_epochs=2000)
```

---

### Inspecting convergence

```python
import matplotlib.pyplot as plt

model = LinearRegression(solver='gd', lr=0.01, n_epochs=500)
model.fit(X_train, y_train)

# plot the cost curve to verify gradient descent is converging
plt.plot(model.costs_)
plt.xlabel('Epoch')
plt.ylabel('Cost (MSE)')
plt.title('Learning curve')
plt.show()
```

> If the cost curve is decreasing smoothly and flattening out, gradient descent is working correctly.
> If it oscillates or increases, the learning rate `lr` is too high — reduce it.

---

### Early stopping behaviour

```python
# tol controls early stopping — training stops when cost improvement < tol
model = LinearRegression(solver='gd', lr=0.01, n_epochs=10000, tol=1e-6)
model.fit(X_train, y_train)
# → "Converged at epoch 243"  (stops early instead of running all 10000 epochs)
```

---

### Using in a full GlassBox pipeline

```python
from cleaner.simple_imputer import SimpleImputer
from cleaner.standard_scaler import StandardScaler
from models.linear_regression import LinearRegression
import numpy as np

# raw data with missing values
X_raw = np.array([[1.0, np.nan],
                  [2.0, 3.0  ],
                  [np.nan, 4.0],
                  [4.0, 5.0  ]])
y = np.array([3.0, 5.0, 7.0, 9.0])

# step 1 — impute missing values
imputer = SimpleImputer()
X_imputed = imputer.fit_transform(X_raw)

# step 2 — scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# step 3 — train model
model = LinearRegression(solver='gd', lr=0.1, n_epochs=1000)
model.fit(X_scaled, y)

# step 4 — predict on new data (apply same pipeline)
X_new = np.array([[3.0, 6.0]])
X_new_imputed = imputer.transform(X_new)
X_new_scaled  = scaler.transform(X_new_imputed)
print(model.predict(X_new_scaled))
```

> Always apply `transform()` — never `fit_transform()` — on test or new data. The imputer and scaler must use statistics learned from training data only.

---

### Accessing learned weights

```python
model = LinearRegression(solver='gd')
model.fit(X_train, y_train)

bias      = model.w_[0]     # intercept (absorbed into w_)
weights   = model.w_[1:]    # one weight per feature

print(f"Bias: {bias:.4f}")
for i, w in enumerate(weights):
    print(f"w{i+1}: {w:.4f}")
```
