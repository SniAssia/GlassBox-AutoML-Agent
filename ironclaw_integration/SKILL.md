---
name: AutoFit
description: Runs a full AutoML pipeline (EDA, cleaning, training) on a local CSV file to predict a target column.
author: GlassBox
version: 0.1.0

# Defines how the skill is executed.
# 'exec' runs a command-line tool.
runtime: exec

# The command to run. The arguments in {curly_braces} will be
# filled in by the agent based on the user's request.
#
# Optional third argument: a JSON string or a path to a JSON config file.
# If omitted, the tool returns an inspection report + available options.
# If provided, the tool runs preprocessing + hyperparameter search.
# Example (JSON inline):
#   {"seed":42,"preprocess":{"imputer":{"method":"mean"}},"search":{"strategy":"grid","cv":{"n_splits":3},"models":[{"name":"logistic_regression","space":{"lr":[0.01,0.1],"n_epochs":[100,200]}}]}}

command: C:\Users\salim\Documents\Projects\AI\GlassBox-AutoML-Agent\.venv\Scripts\python.exe autofit.py {file_path} {target_variable} {config_json}

# Defines the parameters the agent needs to extract from the user's prompt.
parameters:
  - name: file_path
    type: string
    description: The local path to the CSV file for training the model.
    required: true
  - name: target_variable
    type: string
    description: The name of the column in the CSV file that the model should learn to predict.
    required: true
  - name: config_json
    type: string
    description: Optional JSON config (or a path to a JSON file) that specifies preprocessing, candidate models, and search strategy.
    required: false
---

## AutoFit Skill

This skill provides a tool to automatically build a machine learning model from a CSV file.

It performs the following steps:
1.  **Automated EDA**: Analyzes the data to understand its statistical properties.
2.  **Preprocessing**: Cleans the data by imputing missing values and scaling features.
3.  **Modeling**: Uses the GlassBox-AutoML library to find the best model and hyperparameters.
4.  **Reporting**: Returns a JSON report with the results.
