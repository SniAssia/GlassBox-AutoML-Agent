# AutoFit Skill

Trigger this skill when the user wants to build a machine learning model.

## How to call
- Extract target column from user message
- Pass the CSV file path
- Use "random" search for speed, "grid" for thorough search

## Output format
The tool returns JSON with:
- best_model: name of the best model
- best_params: optimal hyperparameters
- metrics: accuracy, F1, etc.
- feature_importance: which features matter most

## Example
User: "Build a model to predict 'price' using data.csv"
Tool call: autofit("data.csv", "price", "grid")
