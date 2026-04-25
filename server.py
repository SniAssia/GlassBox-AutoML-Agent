from fastmcp import FastMCP
from autofit import run_autofit

mcp = FastMCP("autofit_agent")

@mcp.tool() # <--- ADD THIS
def run_autofit_tool(csv_path: str, target_column: str, cv_splits: int = 5, search_strategy: str = "grid", random_iter: int = 10):
    """
    Runs an AutoML pipeline on a CSV file.
    :param csv_path: Path to the CSV data.
    :param target_column: The name of the column to predict.
    """
    results = run_autofit(
        csv_path,
        target_column,
        config={
            "cv_splits": cv_splits,
            "search_strategy": search_strategy,
            "random_iter": random_iter,
        }
    )
    return results

if __name__ == "__main__":
    mcp.run(transport="stdio")
