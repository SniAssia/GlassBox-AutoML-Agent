from pathlib import Path

from autofit import run_autofit


def test_autofit_flow_returns_complete_report():
    report = run_autofit(Path("data/data.csv"), "Class", config={"cv_splits": 3})

    assert report["status"] == "success"
    assert report["input"]["target_column"] == "Class"
    assert "dataset_summary" in report
    assert "eda_summary" in report
    assert "preprocessing_summary" in report
    assert "search_summary" in report
    assert "best_model" in report
    assert "explainability" in report
    assert report["best_model"]["name"]
