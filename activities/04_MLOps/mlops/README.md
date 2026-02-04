# MLOps - Experiment Tracking & Model Management

## 📚 Overview

This module covers **MLOps fundamentals** including experiment tracking, model versioning, and artifact logging using MLflow and Weights & Biases (W&B).

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Track experiments with **MLflow**
- ✅ Log metrics, parameters, and artifacts
- ✅ Use **Weights & Biases** for experiment visualization
- ✅ Version and register models
- ✅ Compare experiments across runs

---

## 📂 Module Structure

```
mlops/
├── README.md (this file)
├── MLOps_Demos.md (Overview guide)
├── mlflow_demo.ipynb
├── wandb_demo.ipynb
├── exercise/
├── solution/
├── artifacts/
├── mlruns/ (MLflow tracking)
└── wandb/ (W&B tracking)
```

---

## 🔄 Recommended Learning Path

### **Part 1: MLflow** (2-3 hours)

1. Review `MLOps_Demos.md` for concepts
2. `mlflow_demo.ipynb`
   - Tracking API
   - Logging parameters and metrics
   - Model registry
   - Artifact storage

### **Part 2: Weights & Biases** (2 hours)

3. `wandb_demo.ipynb`
   - Dashboard visualizations
   - Hyperparameter sweeps
   - Team collaboration

### **Part 3: Practice** (1-2 hours)

4. Complete exercises in `exercise/` folder

---

## 🔍 Tools Covered

| Tool | Strengths | Best For |
|------|-----------|----------|
| **MLflow** | Open-source, self-hosted | Full control, on-prem |
| **W&B** | Rich visualizations, collaboration | Team projects, research |

---

## 🛠️ Technical Requirements

```python
mlflow, wandb, scikit-learn
```

---

## 🔗 Related Modules

- **Prerequisites**: [Ensemble Methods](../../01_Classical_ML/ensemble/), [Cross Validation](../../01_Classical_ML/cross_validation/)
- **Related**: [Model Drift](../../01_Classical_ML/drift/)

---

*Module Difficulty: Intermediate*
