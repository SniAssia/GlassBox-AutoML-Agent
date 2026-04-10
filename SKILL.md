---
name: GlassBox AutoML
endpoint: https://musical-sniffle-g479jrrwr49r2vqjx-8000.app.github.dev/run-automl
method: POST
---

# GlassBox AutoML
This tool allows the IronClaw agent to train machine learning models using your local FastAPI backend.

## Parameters
- `csv_base64`: The dataset in Base64 format.
- `target_column`: The name of the column to predict.
- `task_type`: Either "classification" or "regression".