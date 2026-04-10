---
name: GlassBox_AutoML
endpoint: https://musical-sniffle-g479jrrwr49r2vqjx-8000.app.github.dev/run-automl
method: POST
---

# GlassBox AutoML
This tool allows IronClaw to run your local AutoML pipeline.

## Required Parameters
- `csv_base64`: Your data as a base64 string.
- `target_column`: The name of the column to predict.
- `task_type`: Use "classification" or "regression".