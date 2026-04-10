# Logistic Regression — Theory Notes

*author: @Soufiane AIT LHADJ*

---

## 1. From Linear to Logistic

Linear regression predicts a continuous value $f(\mathbf{x}) = \mathbf{w}^T\mathbf{x}$ which can range from $-\infty$ to $+\infty$. For classification we need an output in $(0, 1)$ to represent a **probability**. We achieve this by wrapping the linear output with the **sigmoid function**.

---

## 2. The Sigmoid Function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- Output is always in $(0, 1)$ — interpretable as a probability
- $\sigma(0) = 0.5$ — at zero the model is maximally uncertain
- $\sigma(z) \to 1$ as $z \to +\infty$
- $\sigma(z) \to 0$ as $z \to -\infty$

**The model prediction becomes:**

$$\hat{p} = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1 + e^{-\mathbf{w}^T\mathbf{x}}}$$

This is interpreted as: *"the probability that sample $\mathbf{x}$ belongs to class 1."*

**Useful derivative** (used in gradient derivation):

$$\sigma'(z) = \sigma(z)(1 - \sigma(z))$$

**Proof:**

$$\frac{d}{dz}\frac{1}{1+e^{-z}} = \frac{e^{-z}}{(1+e^{-z})^2} = \frac{1}{1+e^{-z}} \cdot \frac{e^{-z}}{1+e^{-z}} = \sigma(z)(1-\sigma(z))$$

**Numerical stability:** For very large or very small $z$, `np.exp(-z)` can overflow. Always clip $z$ before applying:

```python
z = np.clip(z, -500, 500)
return 1 / (1 + np.exp(-z))
```

---

## 3. The Cost Function — Binary Cross-Entropy

### Why not MSE?

We cannot use MSE for logistic regression because composing it with the sigmoid makes the loss **non-convex** — gradient descent would get stuck in local minima.

### Derivation from Maximum Likelihood

We model the prediction as a **Bernoulli probability**. Both cases ($y=1$ and $y=0$) can be written in one compact expression:

$$P(y_i | \mathbf{x}_i) = \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}$$

Check:
- When $y_i = 1$: gives $\hat{p}_i$ 
- When $y_i = 0$: gives $1 - \hat{p}_i$ 

Assuming samples are independent, the **joint likelihood** of all $m$ samples is:

$$L(\mathbf{w}) = \prod_{i=1}^{m} \hat{p}_i^{y_i}(1-\hat{p}_i)^{1-y_i}$$

We maximize the **log-likelihood** (log is monotonic — same optimum, easier to work with):

$$\log L(\mathbf{w}) = \sum_{i=1}^{m} \left[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]$$

Since gradient descent **minimizes**, we flip the sign and normalize by $m$:

$$\boxed{C(\mathbf{w}) = -\frac{1}{m}\sum_{i=1}^{m}\left[y_i\log(\hat{p}_i) + (1-y_i)\log(1-\hat{p}_i)\right]}$$

This is the **negative log-likelihood**, also called **binary cross-entropy**.

### Intuition

The loss heavily penalizes **confident wrong predictions**:

| $y_i$ | $\hat{p}_i$ | cost | meaning |
|---|---|---|---|
| 1 | 0.99 | ≈ 0.01 | correct & confident → tiny cost |
| 1 | 0.50 | ≈ 0.69 | uncertain → moderate cost |
| 1 | 0.01 | ≈ 4.60 | wrong & confident → huge cost |
| 0 | 0.01 | ≈ 0.01 | correct & confident → tiny cost |
| 0 | 0.50 | ≈ 0.69 | uncertain → moderate cost |
| 0 | 0.99 | ≈ 4.60 | wrong & confident → huge cost |

**Numerical stability:** $\log(0)$ is undefined. Always clip $\hat{p}$ before taking the log:

```python
p = np.clip(p, 1e-15, 1 - 1e-15)
```

> Note: in Python, `10^(-15)` is **not** $10^{-15}$ — `^` is bitwise XOR. Always use `1e-15`.

---

## 4. Gradient Derivation

We need $\nabla_\mathbf{w} C$ to run gradient descent. We apply the **chain rule**:

$$\frac{\partial C}{\partial \mathbf{w}} = \frac{\partial C}{\partial \hat{p}_i} \cdot \frac{\partial \hat{p}_i}{\partial \mathbf{w}}$$

**Step 1 — Differentiate $C$ with respect to $\hat{p}_i$:**

$$\frac{\partial C}{\partial \hat{p}_i} = -\frac{1}{m}\left[\frac{y_i}{\hat{p}_i} - \frac{1 - y_i}{1 - \hat{p}_i}\right] = -\frac{1}{m} \cdot \frac{y_i - \hat{p}_i}{\hat{p}_i(1-\hat{p}_i)}$$

**Step 2 — Differentiate $\hat{p}_i$ with respect to $\mathbf{w}$ (chain rule + sigmoid derivative):**

$$\frac{\partial \hat{p}_i}{\partial \mathbf{w}} = \sigma'(\mathbf{w}^T\mathbf{x}_i) \cdot \mathbf{x}_i = \hat{p}_i(1-\hat{p}_i) \cdot \mathbf{x}_i$$

**Step 3 — Multiply (chain rule):**

$$\frac{\partial C}{\partial \mathbf{w}} = -\frac{1}{m} \cdot \frac{y_i - \hat{p}_i}{\hat{p}_i(1-\hat{p}_i)} \cdot \hat{p}_i(1-\hat{p}_i) \cdot \mathbf{x}_i$$

The $\hat{p}_i(1-\hat{p}_i)$ terms **cancel** — this is the beautiful cancellation unique to logistic regression:

$$= \frac{1}{m}(\hat{p}_i - y_i)\mathbf{x}_i$$

**Step 4 — Sum over all samples → matrix form:**

$$\boxed{\nabla_\mathbf{w} C = \frac{1}{m} X^T(\hat{\mathbf{p}} - \mathbf{y}) = \frac{1}{m} X^T(\sigma(X\mathbf{w}) - \mathbf{y})}$$

> Notice how similar this is to the linear regression gradient $\frac{1}{m}X^T(X\mathbf{w} - \mathbf{y})$ — the only difference is the sigmoid wrapping $X\mathbf{w}$.

---

## 5. Why No Normal Equation?

In linear regression, setting $\nabla_\mathbf{w} C = 0$ gave a linear equation in $\mathbf{w}$ solvable analytically:

$$\mathbf{w} = (X^TX)^{-1}X^T\mathbf{y}$$

For logistic regression, setting $\nabla_\mathbf{w} C = 0$ gives:

$$X^T\sigma(X\mathbf{w}) = X^T\mathbf{y}$$

The sigmoid wraps $\mathbf{w}$ in a **nonlinear transcendental function** — it is impossible to isolate $\mathbf{w}$ algebraically. There is no closed form solution. Gradient descent is the only option.

---

## 6. Gradient Descent Update Rule

At each iteration:

$$\mathbf{w} := \mathbf{w} - \alpha \nabla_\mathbf{w} C = \mathbf{w} - \frac{\alpha}{m} X^T(\sigma(X\mathbf{w}) - \mathbf{y})$$

Where $\alpha$ is the **learning rate**.

**Cost must always decrease.** If it increases or oscillates:
- Learning rate $\alpha$ is too high → reduce it
- Gradient sign is wrong → check `errors = p_hat - y`, not `y - p_hat`
- Sigmoid is wrong → check for missing minus sign: `1 / (1 + np.exp(-z))`

---

## 7. Decision Boundary

The model predicts class 1 when $\hat{p} \geq \text{threshold}$:

$$\hat{y} = \begin{cases} 1 & \text{if } \sigma(\mathbf{w}^T\mathbf{x}) \geq \text{threshold} \\ 0 & \text{otherwise} \end{cases}$$

- **Lower threshold** → predicts class 1 more aggressively → higher recall, lower precision
- **Higher threshold** → more conservative → lower recall, higher precision

---

## 8. Multi-Class: One-vs-Rest (OvR)

For $K$ classes, we train $K$ independent binary classifiers. Classifier $i$ answers: *"Is this sample class $i$, or not?"*

**Training:** for each class $k$:

$$y_{\text{binary}}^{(k)} = \mathbf{1}[y = k] \qquad \text{(1 for class k, 0 for all others)}$$

Train a separate $\mathbf{w}^{(k)}$ using gradient descent on this binary problem. This gives a weight matrix of shape $(K, n+1)$.

**Prediction:** all $K$ classifiers output a probability. The winning class is:

$$\hat{y} = \arg\max_{k} \hat{p}_k$$

**Example with 3 classes:**

| Classifier | Question | Raw output |
|---|---|---|
| Classifier 0 | Is it class 0? | $\hat{p}_0 = 0.15$ |
| Classifier 1 | Is it class 1? | $\hat{p}_1 = 0.72$ |
| Classifier 2 | Is it class 2? | $\hat{p}_2 = 0.31$ |

Predicted class → **class 1** (highest probability).

**Normalization:** raw OvR probabilities don't sum to 1 — each classifier was trained independently. We normalize each row:

$$\hat{p}_k^{\text{normalized}} = \frac{\hat{p}_k}{\sum_{j=1}^{K} \hat{p}_j}$$

> Normalization does not change which class wins — $\arg\max$ is invariant to positive scaling.

---

## 9. Important Implementation Details

**Bias absorbed into w:**

A column of ones is prepended to $X$ internally so the bias $b$ is absorbed as $w_0$. Never prepend the bias column yourself before calling `fit()` or `predict()` — it will be added twice.

**Labels can be anything:**

The class discovers unique labels automatically via `np.unique(y)`. Labels can be integers, strings, or any comparable type.

**`weights_` is 2D:**

Unlike binary logistic regression where `w` is a 1D vector of shape `(n+1,)`, here `weights_` has shape `(K, n+1)` — one row per class.

**Map indices back to labels:**

`np.argmax` returns the index of the winning class, not the label. Always map back:

```python
indices = np.argmax(probs, axis=1)
return self.classes_[indices]   # map index → original label
```

---

## 10. Comparison: Linear vs Logistic Regression

| | Linear Regression | Logistic Regression |
|---|---|---|
| Task | Regression | Classification |
| Output | $X\mathbf{w}$ (any real value) | $\sigma(X\mathbf{w}) \in (0,1)$ |
| Loss | MSE | Binary cross-entropy |
| Loss origin | Geometric | Maximum likelihood |
| Gradient | $\frac{1}{m}X^T(X\mathbf{w} - \mathbf{y})$ | $\frac{1}{m}X^T(\sigma(X\mathbf{w}) - \mathbf{y})$ |
| Closed form | Yes — normal equation | No — nonlinear in $\mathbf{w}$ |
| Convexity | Yes (quadratic loss) | Yes (log loss) |
| Solver | GD or normal equation | Gradient descent only |
| Multi-class | Native | One-vs-Rest (K classifiers) |

---

## 11. Usage Examples

### Binary classification

```python
from models.logistic_regression import LogisticRegression
import numpy as np

X_train = np.array([[1.0, 2.0],
                    [2.0, 3.0],
                    [3.0, 4.0],
                    [4.0, 5.0]])

y_train = np.array([0, 0, 1, 1])

model = LogisticRegression(lr=0.1, n_epochs=1000)
model.fit(X_train, y_train)

print(model.predict(X_train))          # → [0 0 1 1]
print(model.predict_proba(X_train))    # → probability matrix (m, K)
```

---

### Multi-class classification

```python
from models.logistic_regression import LogisticRegression
import numpy as np

# 3 classes — labels can be integers or strings
X_train = np.random.randn(90, 4)
y_train = np.array(['cat']*30 + ['dog']*30 + ['fish']*30)

model = LogisticRegression(lr=0.1, n_epochs=500)
model.fit(X_train, y_train)

print(model.classes_)           # → ['cat' 'dog' 'fish']
print(model.weights_.shape)     # → (3, 5) — 3 classes, 4 features + bias
print(model.predict(X_train[:3]))
```

---

### Adjusting the threshold

```python
# default threshold = 0.5
model = LogisticRegression(threshold=0.5)

# lower threshold → more aggressive class 1 predictions (higher recall)
model = LogisticRegression(threshold=0.3)

# higher threshold → more conservative (higher precision)
model = LogisticRegression(threshold=0.7)
```

---

### Inspecting learned weights

```python
model.fit(X_train, y_train)

for i, cls in enumerate(model.classes_):
    bias    = model.weights_[i, 0]
    weights = model.weights_[i, 1:]
    print(f'Class {cls}: bias={bias:.3f}, weights={weights}')
```

---

### Using in a full GlassBox pipeline

```python
from cleaner.simple_imputer import SimpleImputer
from cleaner.standard_scaler import StandardScaler
from models.logistic_regression import LogisticRegression
import numpy as np

X_raw   = np.array([[1.0, np.nan], [2.0, 3.0], [np.nan, 4.0], [4.0, 5.0]])
y_train = np.array([0, 0, 1, 1])

# step 1 — impute
imputer = SimpleImputer()
X_imp   = imputer.fit_transform(X_raw)

# step 2 — scale (fit on train only)
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X_imp)

# step 3 — train
model   = LogisticRegression(lr=0.1, n_epochs=1000)
model.fit(X_sc, y_train)

# step 4 — predict on new data (transform only — never fit_transform)
X_new     = np.array([[3.0, 6.0]])
X_new_imp = imputer.transform(X_new)
X_new_sc  = scaler.transform(X_new_imp)
print(model.predict(X_new_sc))
print(model.predict_proba(X_new_sc))
```

> Always apply `transform()` — never `fit_transform()` — on test or new data. The imputer and scaler must use statistics learned from training data only.