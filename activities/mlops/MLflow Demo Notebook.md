# MLflow Demo Notebook

This comprehensive Jupyter notebook provides a **beginner-friendly, hands-on** demonstration of MLflow for teaching MLOps concepts in a machine learning class.

## 📚 Learning Objectives

By completing this notebook, students will learn to:

1. **Experiment Tracking** - Log parameters, metrics, and models systematically
2. **Autologging** - Automatically capture ML metadata with minimal code
3. **Model Registry & Versioning** - Manage model lifecycle (Staging → Production)
4. **Hyperparameter Tuning** - Compare multiple runs with nested experiments
5. **Artifact Logging** - Save plots, confusion matrices, and data files
6. **Model Loading** - Retrieve and use previously trained models
7. **Querying Experiments** - Programmatically search and filter runs
8. **Model Serving** - Deploy models for predictions (advanced topic)
9. **Best Practices** - Industry-standard MLOps workflows
10. **Troubleshooting** - Common issues and solutions

## Prerequisites

- Python 3.7 or higher
- Jupyter Notebook or JupyterLab

## Installation

**It is highly recommended to run these commands in a new virtual environment** to avoid conflicts with other projects.

### Create a Virtual Environment (Recommended)

```bash
# Create a new virtual environment
python -m venv .venv

# Activate it (macOS/Linux)
source .venv/bin/activate

# Or activate it (Windows)
.venv\Scripts\activate
```

### Install Required Packages

Install the required packages using pip or uv:

```bash
# Using pip
pip install mlflow scikit-learn matplotlib

# Or using uv (faster)
uv pip install mlflow scikit-learn matplotlib
```

## Running the Notebook

### Quick Start

1. **Open the notebook** in VS Code, Jupyter Lab, or Jupyter Notebook

2. **Start the MLflow UI** (in a terminal in the notebook's directory):
   ```bash
   mlflow ui
   ```

3. **Open the MLflow UI** in your browser at: **`http://127.0.0.1:5000`**
   - ⚠️ Use `127.0.0.1` NOT `localhost` (newer MLflow security requirements)

4. **Run the notebook cells sequentially** and refresh the MLflow UI to see results

### Recommended Workflow

- Keep the MLflow UI open in a browser tab while working through the notebook
- Run cells one at a time to understand each concept
- Check the MLflow UI after each major section to see what was logged
- Follow the checklist at the beginning of the notebook

## Navigating the MLflow UI

The MLflow UI provides a visual interface to explore your experiments:

### What You'll Find:

1. **Experiments Page** - List of all experiments
2. **Runs Table** - All runs within an experiment with sortable columns
3. **Run Details** - Click any run to see:
   - Parameters logged
   - Metrics with charts
   - Artifacts (models, plots, files)
   - Tags and metadata
4. **Compare Runs** - Select multiple runs and click "Compare" to see side-by-side comparisons
5. **Model Registry** - Registered models with versions and stages

### Tips:

- Use filters to find specific runs
- Sort by metrics to find best models
- Create scatter plots to visualize parameter vs metric relationships
- Download artifacts directly from the UI

## Dataset

The notebook uses the **Iris dataset** from scikit-learn, a classic dataset for classification tasks:
- **150 samples** of iris flowers
- **4 features**: sepal length, sepal width, petal length, petal width
- **3 classes**: setosa, versicolor, virginica

This simple dataset allows students to focus on learning MLflow concepts without complexity.

## Key Features of This Notebook

✅ **Beginner-friendly** - Assumes no prior MLflow knowledge  
✅ **Comprehensive** - Covers all major MLflow features  
✅ **Hands-on** - Real code examples you can run  
✅ **Best practices** - Industry-standard workflows  
✅ **Troubleshooting** - Solutions to common issues  
✅ **Self-contained** - All instructions included  
✅ **Visual** - Includes plots and artifacts logging  

## What's New (2025 Update)

- ✨ Added autologging examples (easiest way to use MLflow)
- ✨ Model loading and inference section
- ✨ Querying experiments programmatically
- ✨ Comprehensive best practices guide
- ✨ Troubleshooting section with common issues
- ✨ Fixed MLflow UI access issues (127.0.0.1 vs localhost)
- ✨ Enhanced hyperparameter tuning with nested runs
- ✨ Real artifact examples (confusion matrices, JSON, reports)
- ✨ Quick reference cheat sheet

## For Instructors

This notebook is designed to be:
- **Taught in 60-90 minutes** with live coding
- **Self-study friendly** for students to complete on their own
- **Adaptable** - Easy to modify for specific use cases
- **Production-ready** - Teaches real-world MLOps practices

Students should have basic Python and machine learning knowledge before starting.

## Model Serving (Advanced)

### Option 1: Command Line (for testing)

In a **new terminal** in the notebook directory:

```bash
mlflow models serve -m "models:/IrisClassifier/Production" -p 5001 --env-manager=local
```

Then make predictions:

```bash
curl -X POST -H "Content-Type:application/json" \
  --data '{"dataframe_split": {"columns": ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"], "data": [[5.1, 3.5, 1.4, 0.2]]}}' \
  http://127.0.0.1:5001/invocations
```

### Option 2: Python API (recommended in notebook)

The notebook demonstrates loading and using models directly in Python, which is more practical for most use cases.

## Troubleshooting

### Issue: Blank screen on MLflow UI
**Solution:** Use `http://127.0.0.1:5000` instead of `http://localhost:5000`

### Issue: "pyenv binary not found" error
**Solution:** Add `--env-manager=local` flag when serving models

### Issue: Cannot find runs
**Solution:** Make sure `mlflow ui` is running in the same directory as the `mlruns` folder

### Issue: Large repository size from MLflow runs
**Solution:** Add `mlruns/` to your `.gitignore` file to avoid committing experiment data to Git. The `mlruns` folder can become very large with model artifacts.

**Example `.gitignore`:**
```
mlruns/
*.pyc
__pycache__/
.venv/
```

For more troubleshooting tips, see the **Troubleshooting section** in the notebook.



