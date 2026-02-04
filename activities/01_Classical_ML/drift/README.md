# Model Drift & Retraining

## 📚 Overview

This module covers **model drift detection** and **automated retraining strategies**. Learn to monitor deployed models and maintain performance over time as data distributions change.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Understand **concept drift** vs. **data drift**
- ✅ Calculate **Population Stability Index (PSI)** for distribution monitoring
- ✅ Apply statistical tests for drift detection (KS-test, Wasserstein distance)
- ✅ Implement **ADWIN** for streaming drift detection
- ✅ Design **threshold sensitivity analysis** for drift alerts
- ✅ Build **automated retraining pipelines**

---

## 📂 Module Structure

```
drift/
├── README.md (this file)
├── model_drift_demo.ipynb (Comprehensive demo)
├── model_drift_and_retraining.ipynb (Full pipeline)
├── 01_psi_calculation_visualization.ipynb
├── 02_ks_wasserstein_comparison.ipynb
├── 03_adwin_streaming_detection.ipynb
├── 04_threshold_sensitivity_analysis.ipynb
└── 05_retraining_window_experiments.ipynb
```

---

## 🔄 Recommended Learning Path

### **Part 1: Understanding Drift** (2...3 hours)

1. `model_drift_demo.ipynb` - Conceptual overview and visualization
2. `01_psi_calculation_visualization.ipynb` - PSI deep dive

### **Part 2: Detection Methods** (2-3 hours)

3. `02_ks_wasserstein_comparison.ipynb` - Statistical tests comparison
4. `03_adwin_streaming_detection.ipynb` - Online/streaming detection

### **Part 3: Retraining Strategies** (2-3 hours)

5. `04_threshold_sensitivity_analysis.ipynb` - Alert tuning
6. `05_retraining_window_experiments.ipynb` - Window strategies
7. `model_drift_and_retraining.ipynb` - Complete pipeline

---

## 🔍 Methods Covered

| Method | Type | Description |
|--------|------|-------------|
| **PSI** | Statistical | Population Stability Index for distribution shift |
| **KS-Test** | Statistical | Kolmogorov-Smirnov for distribution comparison |
| **Wasserstein** | Statistical | Earth mover's distance between distributions |
| **ADWIN** | Streaming | Adaptive windowing for concept drift |
| **Chi-Square** | Statistical | Categorical feature drift detection |

---

## 🛠️ Technical Requirements

```python
numpy, pandas, scipy, scikit-learn, river (for ADWIN)
```

---

## 🔗 Related Modules

- **Previous**: Model training modules
- **Next**: [MLOps](../../04_MLOps/mlops/)

---

*Module Difficulty: Intermediate to Advanced*  