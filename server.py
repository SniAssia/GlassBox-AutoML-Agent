<<<<<<< HEAD
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
=======
from mcp.server.fastmcp import FastMCP
import numpy as np
from autofit import run_autofit

from Orchestrator.GridSearch import GridSearch
from Orchestrator.KFoldCV import KFoldCV
from Orchestrator.Orchestrator import GlassBoxAutoML
from Orchestrator.RandomSearch import RandomSearch
from models.KNN import KNearestNeighbors
from models.decision_tree import DecisionTree
from models.linear_regression import LinearRegression
from models.logistic_regression import LogisticRegression
from models.naive_bayes import GaussianNaiveBayes
from models.random_forest import RandomForest

# Initialize the server with a name
mcp = FastMCP("AutoML Server")

@mcp.tool()
def run_autofit_tool(csv_path, target_column, cv_splits=5, search_strategy="random", random_iter=1):   
        
    config={
        "cv_splits": cv_splits,
        "search_strategy": search_strategy,
        "random_iter": random_iter,
    }

    report = run_autofit(csv_path, target_column, config)

    # report = "This is working! Received the following parameters:\n"
    # report += f"CSV Path: {csv_path}\n"
    # report += f"Target Column: {target_column}\n"
    # report += f"CV Splits: {cv_splits}\n"
    # report += f"Search Strategy: {search_strategy}\n"
    # report += f"Random Iterations: {random_iter}\n"

    return report
>>>>>>> a9930c2 (b)

if __name__ == "__main__":
    mcp.run(transport="stdio")
