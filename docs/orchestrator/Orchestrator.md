# Optimization Module – GlassBox AutoML

## 🔷 1. Overview

This module is responsible for **automatically selecting the best model configuration** through hyperparameter tuning.

### Objective

Find the **best hyperparameters** such that the model:

* Performs well on training data
* **Generalizes** to unseen data

---

## 🔷 2. High-Level Workflow

1. Receive **cleaned data** from the preprocessing module
2. Generate multiple **hyperparameter configurations**
3. Evaluate each configuration using **K-Fold Cross-Validation**
4. Select the configuration with the **best average score**

---

## 🔷 3. Design Principles (OOP)

* Each component has a **single responsibility**
* Components are **modular and interchangeable**
* The **Orchestrator coordinates everything**

---

## 🔷 4. Module Structure

### A. `KFoldCV.py` — Cross-Validation Engine

#### Role

Splits the dataset into **K folds** to evaluate model robustness.

#### Why

* Prevent overfitting
* Ensure reliable performance estimation

#### How it works

* Optionally shuffle data
* Split into K folds
* For each fold:

  * Use 1 fold for validation
  * Use K−1 folds for training

#### Output

```python
(train_indices, validation_indices)
```

---

### B. `HyperParameterSearch.py` — Abstract Base Class

#### Role

Defines a **common interface** for search strategies.

#### Why

Allows interchangeable use of:

* Grid Search
* Random Search

#### Key Method

```python
search(X, y, cv)
```

#### Returns

* Best score
* Best hyperparameters

---

### C. `GridSearch.py` — Exhaustive Search

#### Role

Tests **all possible hyperparameter combinations**.

#### Example

```python
{
  "k": [3, 5, 7],
  "distance_metric": ["euclidean", "manhattan"]
}
```

#### Behavior

Evaluates:

```
(3, euclidean), (3, manhattan), (5, euclidean), ...
```

#### Pros

* Guarantees best solution within grid

#### Cons

* Computationally expensive

---

### D. `RandomSearch.py` — Stochastic Search

#### Role

Tests a **random subset of hyperparameters**.

#### How

* Random sampling from parameter space
* Controlled by `n_iter`

#### Pros

* Faster
* Efficient for large search spaces

#### Cons

* No guarantee of optimal solution

---

### E. `Orchestrator.py` — Pipeline Controller

#### Role

Central component coordinating the pipeline.

#### Responsibilities

* Receive cleaned data
* Call search strategy
* Use cross-validation
* Return final results

#### Core Logic

```python
best_score, best_params = search_strategy.search(X, y, cv)
```

#### Output

* Best score
* Best hyperparameters

---

## 🔷 5. Pipeline Flow

```
Cleaner → Orchestrator → Search Strategy → KFoldCV → Model → Score
```

### Detailed Execution

1. Orchestrator receives `(X_clean, y_clean)`
2. Calls search strategy (Grid or Random)
3. For each hyperparameter set:

   * For each fold:

     * Train model
     * Evaluate using `score()`
4. Compute **average score**
5. Keep the best configuration

---

## 🔷 6. Important Concept: `score()`

### Role

Measures model performance.

### Example (Classification)

```python
accuracy = correct_predictions / total_predictions
```

### Why important

* Used to **compare models**
* Every model must implement:

```python
fit(), predict(), score()
```

---

## 🔷 7. Final Summary

* KFoldCV → robust evaluation
* GridSearch → exhaustive search
* RandomSearch → efficient exploration
* Orchestrator → pipeline control

### Final Output

* Best hyperparameters
* Best validation score

---

## 🔷 8. One-Sentence Summary

> This module automates hyperparameter tuning by combining search strategies with cross-validation to select the best-performing model configuration.
