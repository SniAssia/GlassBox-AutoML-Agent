import argparse
import json

from autofit import run_autofit


def main():
    parser = argparse.ArgumentParser(description="Run the GlassBox AutoFit pipeline.")
    parser.add_argument("--csv", required=True, dest="csv_path", help="Path to the CSV dataset.")
    parser.add_argument("--target", required=True, dest="target_column", help="Target column name.")
    parser.add_argument("--cv-splits", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--search-strategy", choices=["grid", "random"], default="grid")
    parser.add_argument("--random-iter", type=int, default=10)
    args = parser.parse_args()

    report = run_autofit(
        args.csv_path,
        args.target_column,
        config={
            "cv_splits": args.cv_splits,
            "search_strategy": args.search_strategy,
            "random_iter": args.random_iter,
        },
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
