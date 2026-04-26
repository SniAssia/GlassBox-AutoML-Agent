# Automated EDA (The Inspector) – GlassBox AutoML

## 🔷 1. Overview

The Automated Exploratory Data Analysis (EDA) module acts as the "Inspector" for GlassBox-AutoML. It performs a non-destructive audit of raw datasets, providing essential context to the Agent without relying on heavy external libraries like Pandas or Scikit-Learn.

### Objective

Automatically detect, compute, and expose underlying patterns in data to guide subsequent preprocessing steps.

---

## 2. Core Components

### A. Statistical Profiling (`statistics.py`)

A manual implementation of core statistical measures tailored to understand feature distributions.
- **Mean & Median**: Central tendency measures.
- **Mode**: Most frequent values for categorical insights.
- **Standard Deviation**: Data spread and volatility.
- **Skewness & Kurtosis**: Measures of tail behavior and distribution shapes.

### B. Association & Collinearity (`association.py`)

- **Pearson Correlation Matrix**: Built from scratch to compute linear correlations between variables.
- Detects collinearity logic to ensure models do not suffer from duplicated or redundant signals.

### C. Outlier Detection (`iqr.py`)

Implements Interquartile Range (IQR) capabilities.
- Recognizes the 25th (Q1) and 75th (Q3) percentiles.
- Flags data points beyond `1.5 * IQR` as outliers to either ignore or cap.

### D. Auto-Typing (`auto-typing.py`)

Automatically distinguishes and categorizes column structures.
- **Numerical**: Continuous or discrete numbers.
- **Categorical**: Non-numeric strings or distinct objects.
- **Boolean**: True/False or binary logic.
- Crucial for pipeline routing (e.g., Categorical fields go to One-Hot Encoding, Numerical fields to Standard Scaler).

### E. Report Builder (`report_builder.py`)

Gathers all derived insights into a standardized format (usually JSON) that is consumed directly by the Agent.

---

## 3. High-Level Workflow

1. Data ingestion.
2. `auto-typing.py` runs to route fields.
3. `statistics.py` gathers numerical distributions.
4. `association.py` produces the feature correlation matrix.
5. `iqr.py` detects outliers.
6. `report_builder.py` unifies the summary.
