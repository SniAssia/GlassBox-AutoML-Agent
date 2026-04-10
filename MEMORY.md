# Workspace Memory

## Project

This workspace contains `GlassBox-AutoML-Agent`, a scratch-built AutoML toolkit for CSV-based tabular machine learning.

## Main entrypoint

The primary execution path is:

- `autofit.py`

The preferred command-line wrapper for interactive agent use is:

- `python .\\iron_claw_agent\\run_autofit_cli.py --csv <csv_path> --target <target_column>`

## Current capabilities

- infer task type from target column
- run EDA summary
- preprocess mixed tabular data
- select a model with cross-validation
- return JSON output
- provide a simple explainability summary

## Example prompts

- `Build a model to predict survived from data/titanic.csv`
- `Train a model for Exam_Score using data/StudentPerformanceFactors.csv`
- `Run AutoML on data/data.csv with target Class`

## Important caveats

- Outputs are currently strongest for local CSV workflows.
- Classification outputs use `cv_score`.
- Regression outputs use `cv_mse` in the final report.
- Train metrics are still train-set metrics, not holdout metrics.
