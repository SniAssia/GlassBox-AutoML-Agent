# Evaluation Suite – GlassBox AutoML

## 🔷 1. Overview

The Evaluation Suite guarantees that models are properly assessed using scratch-built, transparent metrics covering both Classification and Regression tasks. It focuses on interpretability and robust performance measurement.

### Objective

Compute critical performance indicators and package them into interpretable responses for the Agent.

---

## 🔷 2. Classification Metrics (`metrics_classification.py`, `confusion.py`, `roc_auc.py`)

Used whenever the model predicts a discrete category.

- **Accuracy**: Total correct predictions over total instances.
- **Precision**: True Positives / (True Positives + False Positives). Answers "Of all positive predictions, how many were right?"
- **Recall**: True Positives / (True Positives + False Negatives). Answers "Of all actual positive instances, how many did we catch?"
- **F1-Score**: Harmonic mean of Precision and Recall.
- **Confusion Matrix**: A scratch-built table summarizing TP, FP, TN, and FN.
- **ROC AUC**: Evaluates model distinguishability power across thresholds.

---

## 🔷 3. Regression Metrics (`metrics_regression.py`)

Used whenever the model predicts a continuous value.

- **Mean Absolute Error (MAE)**: Average absolute difference between predicted and true values. Less sensitive to outliers.
- **Mean Squared Error (MSE)**: Average squared difference. Highly penalizes large errors.
- **R² Score (Coefficient of Determination)**: The proportion of variance in the dependent variable predictable from the independent variables. Perfect score is 1.0.

---

## 🔷 4. Explainability & Reporting (`explainability.py`, `report_formatter.py`)

- **Explainability**: Logic to unpack feature importance, highlighting the primary drivers for a prediction (e.g., evaluating which variable contributes most to a tree split).
- **Report Formatter**: Packages the evaluation metrics and evaluation findings into clean, easily readable structures designed to empower Agent language models when explaining outcomes to a user.

---

## 🔷 5. Pipeline Flow

1. Model triggers `predict(X_val)`.
2. Actuals (`y_val`) and predictions are compared.
3. Engine selects Classification or Regression metric routines based on Task type.
4. Scores are stored in an Evaluation Report to feed the cross-validation orchestrator or ultimate user display.
