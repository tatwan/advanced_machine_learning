# AutoML & Low-Code Machine Learning

## 📚 Overview

This module covers **automated machine learning (AutoML)** using various libraries. Learn to rapidly build and compare models with minimal code using PyCaret, H2O, MLJAR, AutoGluon, and FLAML.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Use **PyCaret** for end-to-end ML workflows
- ✅ Apply **H2O AutoML** for automatic model selection
- ✅ Build models with **MLJAR AutoML** (explain/perform/compete modes)
- ✅ Leverage **AutoGluon** for tabular data
- ✅ Use **FLAML** for efficient hyperparameter tuning
- ✅ Compare AutoML libraries for different use cases

---

## 📂 Module Structure

```
automl/
├── README.md (this file)
├── automl_libraries_showcase.md (Library comparison guide)
├── 1.pycaret_demo.ipynb
├── 2.h2o_automl_demo.ipynb
├── 3.mljar_automl.ipynb
├── 4.autogluon_demo.ipynb
├── 5.h2o_flaml_autogluon.ipynb (Comparison notebook)
├── credit_data/
├── mljar_compete/ (Competition mode outputs)
├── mljar_explain/ (Explainability mode outputs)
├── mljar_perform/ (Performance mode outputs)
└── archive/
```

---

## 🔄 Recommended Learning Path

### **Part 1: Introduction to AutoML** (3-4 hours)

1. Review `automl_libraries_showcase.md` for overview
2. `1.pycaret_demo.ipynb` - Low-code ML with PyCaret
3. `2.h2o_automl_demo.ipynb` - H2O's AutoML capabilities

### **Part 2: Alternative Libraries** (3-4 hours)

4. `3.mljar_automl.ipynb` - MLJAR's different modes
5. `4.autogluon_demo.ipynb` - AWS AutoGluon

### **Part 3: Comparison** (2 hours)

6. `5.h2o_flaml_autogluon.ipynb` - Side-by-side comparison

---

## 🔍 Libraries Covered

| Library | Strengths | Best For |
|---------|-----------|----------|
| **PyCaret** | Easy API, full pipeline | Rapid prototyping, beginners |
| **H2O AutoML** | Distributed, robust | Large datasets, production |
| **MLJAR** | Explainability, reports | Interpretable models, reports |
| **AutoGluon** | State-of-art accuracy | Competitions, tabular data |
| **FLAML** | Fast, low resource | Resource-constrained, tuning |

---

## 🛠️ Technical Requirements

```python
pycaret, h2o, mljar-supervised, autogluon, flaml
```

**Note**: Some libraries have conflicting dependencies. Consider using separate environments.

---

## 🔗 Related Modules

- **Prerequisites**: Basic ML concepts
- **Next**: [Ensemble Methods](../ensemble/), [Hyperparameter Optimization](../optimization/)

---

*Module Difficulty: Beginner to Intermediate*  
*Estimated Time: 8-10 hours total*
