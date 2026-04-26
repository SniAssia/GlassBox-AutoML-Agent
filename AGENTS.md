# GlassBox AutoML Workspace Instructions

## Purpose

This workspace provides a GlassBox AutoML capability for tabular CSV datasets.

When the user asks to:

- build a model from a dataset path and target column
- train a model to predict a target from a CSV
- run AutoML on a CSV file
- explain the result of a trained model

the agent should use the GlassBox AutoML workflow defined in the local skill.

## Preferred workflow

1. Extract:
   - `csv_path`
   - `target_column`
2. Run the local AutoFit CLI:
   - `python .\\iron_claw_agent\\run_autofit_cli.py --csv <csv_path> --target <target_column>`
3. Read the JSON result.
4. Explain:
   - detected task type
   - best model
   - cross-validation performance
   - train metrics
   - top explainability features
   - warnings or caveats

## Behavior rules

- Prefer concise, plain-language summaries over dumping the full JSON unless the user asks for raw output.
- If the dataset path or target column is missing, ask only for the missing piece.
- If the command fails, surface the real error message rather than inventing a diagnosis.
- For regression, explain `cv_mse` instead of the internal negative ranking score.
- For classification, explain `cv_score` as cross-validation accuracy-like selection score.

## Notes

- The AutoFit backend already handles EDA, preprocessing, model selection, evaluation, and explainability.
- The current implementation is best suited to local CSV files inside this workspace.
