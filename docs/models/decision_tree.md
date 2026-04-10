# Decision Tree — Theory Notes
---

## 1. Model Overview

Your Decision Tree implementation is a recursive, binary tree learner that supports both:

- Classification
- Regression

The behavior is controlled by:

$$\text{task} \in \{\text{classification},\ \text{regression}\}$$

Each internal node stores:

- The selected feature index
- A split threshold
- Left and right child nodes

Each leaf node stores a final prediction.

---

## 2. Node Structure

The tree is built from a Node object with these key fields:

- feature: index of the feature used for split
- threshold: split value for numeric comparison
- left, right: child nodes
- prediction: if not None, the node is a leaf
- samples_count: number of samples that reached the node
- impurity: impurity value (field exists, currently not explicitly filled during growth)

Split rule used everywhere in the tree:

$$
\text{go left if } x_j \le t,\quad \text{go right if } x_j > t
$$

---

## 3. Impurity Functions

The model switches impurity according to task.

### Classification: Gini impurity

For class proportions $p_k$ in a node:

$$
G = 1 - \sum_k p_k^2
$$

This is implemented by counting labels with NumPy and applying the formula.

### Regression: Mean Squared Error impurity

For targets $y_1,\dots,y_n$ with mean $\mu$:

$$
\text{MSE-node} = \frac{1}{n}\sum_{i=1}^n (y_i - \mu)^2
$$

This is exactly what mse_impurity computes.

---

## 4. Information Gain Used for Splits

For a candidate split producing left and right child sets:

$$
\text{Gain} = I(\text{parent}) - \left(\frac{n_L}{n}I(L) + \frac{n_R}{n}I(R)\right)
$$

Where:

- $I$ is Gini (classification) or MSE (regression)
- $n$ is parent sample count
- $n_L, n_R$ are left and right sample counts

The best split is the one with maximum gain.

---

## 5. Feature Subsampling (Important)

At every node, the algorithm samples a subset of features before searching thresholds:

$$
\text{feature\_indices} \sim \text{without replacement}
$$

- If n_features is None, it uses all features.
- Otherwise it uses min(n_features, total_features).

This introduces randomness and can reduce overfitting, similar to the feature randomness idea used in Random Forests.

---

## 6. Threshold Search Strategy

For each selected feature:

1. Collect unique values in that feature column.
2. Try each unique value as a threshold.
3. Split data using $\le$ and $>$.
4. Compute gain.
5. Keep the best pair (feature, threshold).

If no valid split is found (for example, all splits put everything on one side), the node becomes a leaf.

---

## 7. Stopping Conditions

A node becomes a leaf when any of the following is true:

1. depth >= max_depth
2. n_samples < min_samples_split
3. The node is pure

Purity definition:

- Classification: only one unique class remains
- Regression: all targets are effectively equal using allclose

---

## 8. Leaf Prediction Rule

When a leaf is created:

- Classification: predict the most frequent label (mode)
- Regression: predict the mean target value

So the model output at a leaf is:

$$
\hat{y}_{\text{leaf}} =
\begin{cases}
\operatorname{mode}(y) & \text{classification}\\
\frac{1}{n}\sum_{i=1}^{n} y_i & \text{regression}
\end{cases}
$$

---

## 9. Recursive Training Flow

Training proceeds top-down:

1. Check stopping criteria.
2. If stop: return leaf.
3. Randomly select candidate features.
4. Find best split by max gain.
5. Recurse on left and right partitions.
6. Return decision node.

This process builds the full binary tree.

---

## 10. Prediction Flow

To predict one sample $x$:

1. Start at root.
2. If current node is a leaf, output prediction.
3. Otherwise compare $x_{feature}$ to threshold.
4. Move left or right and repeat.

For a batch of samples, this traversal is applied sample by sample.

---

## 11. Hyperparameters in This Implementation

- max_depth (default 10): maximum allowed depth
- min_samples_split (default 2): minimum samples required to split
- n_features (default None): number of randomly chosen features per node
- task (classification or regression): selects impurity and leaf rule

Input validation in fit includes:

- X must be 2D
- y must be 1D
- Number of rows in X must match length of y

---

## 12. Practical Notes

- Complexity can become high because every unique value can be tested as a threshold.
- The model naturally handles non-linear decision boundaries.
- The current implementation does not expose random_state, so feature subsampling is stochastic across runs.
- For very deep trees, overfitting risk increases; max_depth and n_features help control it.

---

## 13. Why This Design Makes Sense for GlassBox

This implementation is transparent and educational:

- Clear split criterion via impurity reduction
- Interpretable recursive structure
- One code path for classification and regression
- Controlled complexity with depth and split constraints

It is a strong base model for explainable ML and also a natural building block for ensemble methods.