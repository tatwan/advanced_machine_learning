# Table of Contents

- [Course Content Summary](#course-content-summary)
- [Advanced Machine Learning Class Outline](#advanced-machine-learning-class-outline)
- [Environment Setup](#environment-setup)
  - [Using uv](#using-uv)
  - [Using venv (Python built-in)](#using-venv-python-built-in)
  - [Using conda](#using-conda)
- [Installing Packages](#installing-packages)
  - [Using uv](#using-uv-1)
  - [Using pip](#using-pip)
  - [Using conda](#using-conda-1)
- [Running shell commands from Notebooks](#running-shell-commands-from-notebooks)
- [Getting Started with Git](#getting-started-with-git)
  - [Cloning the Repository](#cloning-the-repository)
  - [Updating the Repository](#updating-the-repository)
  - [Handling Modifications and Conflicts](#handling-modifications-and-conflicts)
- [Python environments in VS Code](#python-environments-in-vs-code)
- [Jupyter Notebooks in VS Code](#jupyter-notebooks-in-vs-code)
- [Data Science in VS Code tutorial](#data-science-in-vs-code-tutorial)
- [Manage Jupyter Kernels in VS Code](#manage-jupyter-kernels-in-vs-code)
- [Quickstart for GitHub Codespaces](#quickstart-for-github-codespaces)

## Course Content Summary

This repository covers advanced machine learning topics organized into the following key areas:

### **Anomaly Detection**
- Statistical methods for outlier detection (Tukey method, Z-score, Modified Z-score)
- Machine learning-based anomaly detection using PyOD (Python Outlier Detection)
- Anomaly detection for both time series and non-time series data
- Multiple algorithm families (proximity-based, clustering, neural networks)

### **Deep Learning & Neural Networks**
- Autoencoders for data reconstruction and dimensionality reduction
- Generative Adversarial Networks (GANs) - DCGAN implementation
- Deep learning for feature learning and representation

### **AutoML & Low-Code ML**
- Automated machine learning using PyCaret
- AutoML libraries comparison (H2O AutoML, AutoGluon, FLAML)
- Low-code approaches to model selection and hyperparameter tuning

### **Model Validation & Cross-Validation**
- Train-test splitting strategies
- K-Fold cross-validation techniques
- Stratified cross-validation for classification
- Group-based cross-validation for dependent data
- Understanding overfitting and generalization

### **Model Drift & Retraining**
- Concept drift vs. data drift detection
- Statistical tests for drift (KS-test, Chi-square, Population Stability Index)
- Automated retraining pipelines and triggers
- Model versioning and lifecycle management

### **Ensemble Methods**
- Bagging and boosting techniques
- Random Forests implementation and tuning
- Gradient Boosting algorithms
- AdaBoost and ensemble stacking
- Encoding categorical variables for ensemble models

### **Model Explainability & Interpretability**
- SHAP (SHapley Additive exPlanations) for global and local interpretability
- LIME (Local Interpretable Model-agnostic Explanations)
- Compliance with regulations (GDPR, fair lending laws)
- Building trust and ensuring fairness in ML models

### **Advanced Linear Models**
- Generalized Linear Models (GLMs) for non-normal distributions
- Generalized Additive Models (GAMs) for non-linear relationships
- Interaction terms and feature interactions
- Link functions and model families (Poisson, Binomial, Gamma)

### **Imbalanced Data Handling**
- Oversampling techniques (Random Oversampling, SMOTE, ADASYN, BorderlineSMOTE)
- Undersampling methods (Random Undersampling, Tomek Links, NearMiss)
- Combined approaches (SMOTETomek, SMOTEENN)
- Ensemble methods for imbalanced learning (Balanced Random Forest, EasyEnsemble)
- Cost-sensitive learning and class weights

### **Linear Models & Regularization**
- Linear regression with feature engineering
- Logistic regression for classification
- Regularization techniques (L1, L2, ElasticNet)
- Feature selection and dimensionality reduction

### **Missing Data Handling**
- Data quality checks and diagnostics
- Univariate imputation methods (pandas and scikit-learn)
- Multivariate imputation techniques (KNN, Iterative Imputer)
- Time series interpolation methods

### **MLOps**
- Experiment tracking using MLflow
- Model versioning and registry
- Artifact logging (models, plots, metrics)
- Model deployment and serving
- Reproducibility and collaboration workflows

### **Marketing Analytics**
- Marketing Mix Modeling (MMM) for budget allocation
- Multi-Touch Attribution (MTA) analysis
- ROI calculation for marketing channels
- Time series modeling for marketing impact

### **Hyperparameter Optimization**
- Grid Search for exhaustive parameter search
- Randomized Search for efficient exploration
- Bayesian optimization using Optuna
- Advanced tuning strategies and best practices

### **Feature Engineering & Preprocessing**
- Advanced feature engineering techniques
- Handling dirty data and data quality issues
- Denoising using machine learning models
- Polynomial features and feature interactions
- Scikit-learn preprocessing pipelines
- Feature-engine transformations

### **Time Series Forecasting**
- Facebook Prophet for trend and seasonality analysis
- Theta method for exponential smoothing
- Automated forecasting with StatsForecast
- Handling holidays and special events
- Multi-step ahead forecasting

> [!Note]
>
> The repo got renamed form `adv_ml_ds` to `advanced_machine_learning`. The old URLs and your existing GitHub repo should all work as is (thanks to GitHub automatic redirects)

# Advanced Machine Learning Class Outline

* Day 1: Advanced Data Management and Model Optimization

* Day 2: Ensemble Methods, and Model Robustness

* Day 3: Model Explainability, and Deployment Considerations

## Environment Setup

Below are instructions for setting up virtual environments using different tools.

### Using uv

1. Install uv if not already installed:
   ```bash
   pip install uv
   ```

2. Create a virtual environment:
   ```bash
   uv venv dev1 --python=3.12
   ```

3. Activate the environment:
   ```bash
   source dev1/bin/activate  # On macOS/Linux
   # or
   dev1\Scripts\activate     # On Windows
   ```

4. Deactivate the environment:
   ```bash
   deactivate
   ```

### Using venv (Python built-in)

1. Create a virtual environment:
   ```bash
   python3.12 -m venv dev1
   ```

2. Activate the environment:
   ```bash
   source dev1/bin/activate  # On macOS/Linux
   # or
   dev1\Scripts\activate     # On Windows
   ```

3. Deactivate the environment:
   ```bash
   deactivate
   ```

### Using conda

1. Create a conda environment:
   ```bash
   conda create -n dev1 python=3.12
   ```

2. Activate the environment:
   ```bash
   conda activate dev1
   ```

3. Deactivate the environment:
   ```bash
   conda deactivate
   ```

## Installing Packages 

### Using uv 

```
uv pip install ipykernel pandas matplotlob scikit-learn seaborn
```

### Using pip

```
pip install ipykernel pandas matplotlob scikit-learn seaborn
```

### Using conda 

```
conda install ipykernel pandas matplotlob scikit-learn seaborn
```

## Running shell commands from Notebooks

Shell commands can be executed within a Jupyter Notebook by prefixing the command with an exclamation mark (`!`). This allows users to interact with the underlying operating system directly from within their notebook environment.

```
!uv pip install ipykernel pandas matplotlob scikit-learn seaborn
```



## Getting Started with Git

### Cloning the Repository

To get a copy of this repository on your local machine:

```bash
git clone https://github.com/tatwan/adv_ml_ds.git
cd adv_ml_ds
```

### Updating the Repository

To update your local copy with the latest changes from the remote repository:

```bash
git pull origin main
```

### Handling Modifications and Conflicts

If you've made local modifications and want to update:

1. **Commit your changes first** (if you want to keep them):
   ```bash
   git add .
   git commit -m "Your commit message"
   git pull origin main
   ```

2. **Stash your changes** (if you want to temporarily save them):
   ```bash
   git stash
   git pull origin main
   git stash pop  # To restore your changes
   ```

3. **If conflicts occur during pull**:
   - Git will notify you of conflicts
   - Edit the conflicted files to resolve conflicts
   - Stage the resolved files:
     ```bash
     git add <resolved_file>
     ```
   - Complete the merge:
     ```bash
     git commit
     ```

4. **Force update** (use with caution, this will overwrite local changes):
   ```bash
   git reset --hard origin/main
   ```

## Python environments in VS Code

Read the [official page](https://code.visualstudio.com/docs/python/environments) on the topic

## Jupyter Notebooks in VS Code

Read the [official page](https://code.visualstudio.com/docs/datascience/jupyter-notebooks) on the topic

## Data Science in VS Code tutorial

Read the [official page](https://code.visualstudio.com/docs/datascience/data-science-tutorial) on the topic

## Manage Jupyter Kernels in VS Code

Read the [official page](https://code.visualstudio.com/docs/datascience/jupyter-kernel-management) on the topic

## Quickstart for GitHub Codespaces

Read the [official page](https://docs.github.com/en/codespaces/quickstart) on the topic

