# GlassBox-AutoML-Agent

## Contributors
- [Salim Qadda](https://github.com/lamseey)
- [Salma Ammari](https://github.com/SalmaAMMARI)
- [Soufiane Ait Lhadj](https://github.com/SoufianeAitlhadj)
- [Assia Snissi](https://github.com/SniAssia)

## AutoFit Scaffold
The project now exposes a top-level `run_autofit()` entrypoint in `autofit.py`.

Example usage:

```python
from autofit import run_autofit

report = run_autofit("data/data.csv", "Class")
```

The IronClaw-facing wrapper lives in `iron_claw_agent/tool_autofit.py` and accepts a JSON payload shaped like `iron_claw_agent/demo_request.json`.

Example CLI-style invocation:

```powershell
Get-Content .\iron_claw_agent\demo_request.json | python .\iron_claw_agent\tool_autofit.py
```

Expected output sections:
- `dataset_summary`
- `eda_summary`
- `preprocessing_summary`
- `search_summary`
- `best_model`
- `explainability`
