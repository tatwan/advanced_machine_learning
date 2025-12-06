# 🧪 MLOps Exercise: Model Comparison with MLflow

In this activity, you will practice the core MLOps skill of **Experiment Tracking**. You will train two different models on the Breast Cancer dataset and use MLflow to compare their performance.

## 📂 Files
- `student_activity.ipynb`: The notebook you will work in.

## 🎯 Objectives
1.  **Initialize MLflow**: Set up an experiment.
2.  **Train Models**: Train a Random Forest and a Gradient Boosting model.
3.  **Log Metadata**: Log parameters (config) and metrics (performance) for each run.
4.  **Compare**: Use the MLflow UI to decide which model is better.

## 🚀 Instructions
1.  Open `student_activity.ipynb`.
2.  Follow the `TODO` comments to complete the code.
3.  Run the cells.
4.  Open the MLflow UI to view your results.
    *   Open a terminal.
    *   Navigate to the exercise folder: `cd activities/mlops/exercise`
    *   Run: `mlflow ui`

## 💡 Hints
-   Remember to use `with mlflow.start_run():` to create a run context.
-   Use `mlflow.log_param("name", value)` for inputs.
-   Use `mlflow.log_metric("name", value)` for outputs.
