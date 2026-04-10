# Tools

## AutoFit CLI

Run the local GlassBox AutoML pipeline:

```powershell
python .\iron_claw_agent\run_autofit_cli.py --csv data\titanic.csv --target survived
```

This prints a JSON report with:

- dataset summary
- EDA summary
- preprocessing summary
- search summary
- best model
- explainability

## Raw Python usage

```powershell
python -c "from autofit import run_autofit; import json; print(json.dumps(run_autofit('data/titanic.csv', 'survived'), indent=2))"
```

## Error handling

- If a target column is missing, AutoFit raises a `ValueError`.
- If a model configuration fails during search, the search layer skips it and continues.
- If no model can train successfully, AutoFit reports the failure.
