# GlassBox AutoML Skill

## When to use this skill

Use this skill when the user asks to:

- build or train a model from a CSV
- predict a target column from a dataset path
- run AutoML on a tabular dataset
- explain the best model found by the GlassBox pipeline

Typical user requests:

- `Build a model to predict survived from data/titanic.csv`
- `Train a model for Exam_Score using data/StudentPerformanceFactors.csv`
- `Run AutoML on data/data.csv with target Class`

## Required inputs

Extract these values from the user request:

- `csv_path`
- `target_column`

If one is missing, ask only for the missing value.

## Execution

Run:

```powershell
python .\iron_claw_agent\run_autofit_cli.py --csv <csv_path> --target <target_column>
```

## Expected output

The command returns JSON with:

- `input`
- `dataset_summary`
- `eda_summary`
- `preprocessing_summary`
- `search_summary`
- `best_model`
- `explainability`

## How to answer the user

After reading the JSON:

1. State the detected task type.
2. State the best model.
3. State the main validation result:
   - classification: `cv_score`
   - regression: `cv_mse`
4. Summarize train metrics briefly.
5. Mention the top explainability signals.
6. Mention caveats if needed.

## Output style

Prefer a concise summary like:

- `I trained a classification model for survived from data/titanic.csv.`
- `The best model was random_forest with cross-validation score 0.826.`
- `The strongest signals were fare, age, and pclass.`

Offer the raw JSON only if the user asks for it.
