# Agent Integration – GlassBox AutoML

## 🔷 1. Overview

The GlassBox AutoML library functions as a "Skill" for an IronClaw Agent (or similar Agentic Framework). This module wraps the engine enabling users to perform sophisticated data science tasks through natural language.

### Objective

To provide an interface where an Agent, running inside a secure, WASM-ready environment, can execute the automated EDA, preprocessing, and model search pipeline, and then articulate the results.

---

## 🔷 2. The IronClaw Agent Workflow

1. **Natural Language Trigger**:
   - The User interacts with the Agent: *"Build a model to predict 'Target' using this CSV"*.
   
2. **Tool Execution**:
   - The Agent extracts `csv_path` and `target_column`.
   - The Agent calls the local AutoFit CLI or executes the `AutoFit` tool.
   - Command pattern: `python .\iron_claw_agent\run_autofit_cli.py --csv <csv_path> --target <target_column>`
   
3. **Execution Pipeline**:
   - Behind the scenes, the request triggers the full GlassBox stack: **EDA → Cleaning → Search**.
   - Since the engine has zero dependencies except NumPy, it executes rapidly inside the sandbox (WASM-ready architecture).
   
4. **JSON Reporting**:
   - The pipeline finishes by generating a comprehensive JSON report containing:
     - The detected task type (Classification/Regression).
     - The best model algorithm and its best hyperparameters.
     - Cross-validation performance.
     - Training metrics (MSE, Accuracy, R², etc.).
     - The top features for explainability.
     
5. **Explainability & Narration**:
   - The Agent reads the JSON result and constructs a plain-language summary.
   - For example: *"The model favored 'Age' as the most important feature"* or *"Cross-validation selected a Decision Tree with Accuracy X%."*
   
---

## 🔷 3. Behavioral Rules for Agents

- **Concise Summaries**: The Agent should prefer concise, plain-language summaries over dumping the full JSON to the user (unless explicitly asked).
- **Handling Errors**: If the dataset path or target column is missing, the Agent should ask the user to provide them. If the internal command fails, the Agent surfaces the real error message instead of inventing a diagnosis.
- **Metric Context**:
  - For continuous data (Regression), the Agent explains `cv_mse`.
  - For discrete data (Classification), the Agent explains `cv_score` (like cross-validation accuracy).
  
---

## 🔷 4. File Structure (`iron_claw_agent/`)

- `tool_autofit.py`: Defines the executable logic and wrapping of the GlassBox Auto-ML pipeline into an Agentic tool.
- `run_autofit_cli.py`: The entry-point script executed by the terminal.
- `schemas.py`: Dictates the JSON shapes expected for inputs and outputs.
- `demo_request.json`: Provides an example payload.
