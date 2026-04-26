# KNN Model (Models_Zoo) – GlassBox AutoML

## 🔷 1. What is KNN?

K-Nearest Neighbors is a **supervised learning algorithm** used for:

* Classification
* Regression

### Core Idea

A point is predicted using its **K nearest neighbors**.

---

## 🔷 2. How it works

1. Choose K
2. Compute distances
3. Select K nearest points
4. Aggregate results

---

## 🔷 3. Distance Metrics

### A. Euclidean Distance

```math
d(x, z) = sqrt(Σ (x_i - z_i)^2)
```

* Straight-line distance
* Sensitive to large differences

---

### B. Manhattan Distance

```math
d(x, z) = Σ |x_i - z_i|
```

* Grid-based movement
* More robust to outliers

---

### 🔴 Key Difference (Intuition)

* Euclidean → direct path (diagonal allowed)
* Manhattan → grid path (no diagonal)

Example:

* Euclidean = 5
* Manhattan = 7

---

## 🔷 4. Classification vs Regression in KNN

Same algorithm, different aggregation:

| Task           | Method        |
| -------------- | ------------- |
| Classification | Majority vote |
| Regression     | Mean          |

### Example

**Classification**

```
Neighbors: [0, 1, 1] → Prediction = 1
```

**Regression**

```
Neighbors: [10, 12, 14] → Prediction = 12
```

---

## 🔷 5. Why Distance Matters

KNN depends entirely on:

> “Who are the nearest neighbors?”

Changing distance:

* Changes neighbors
* Changes prediction

---

## 🔷 6. Role of Hyperparameters in KNN

* `k`: number of neighbors
* `distance_metric`: euclidean or manhattan

The optimization module:

* Tests different values
* Selects the best combination using cross-validation
