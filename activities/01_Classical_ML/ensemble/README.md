# Ensemble Methods

## 📚 Overview

This module provides comprehensive coverage of **ensemble learning techniques**, including bagging, boosting, and stacking approaches. You will learn how to combine multiple models to achieve better predictive performance than any single model alone.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Understand the theoretical foundations of ensemble learning
- ✅ Implement **Random Forests** for classification and regression
- ✅ Apply **bagging** techniques to reduce variance
- ✅ Master **boosting** algorithms (Gradient Boosting, AdaBoost, XGBoost)
- ✅ Handle categorical variables with proper encoding techniques
- ✅ Tune hyperparameters for optimal ensemble performance

---

## 📂 Module Structure

```
ensemble/
├── README.md (this file)
├── overview/
│   ├── 1. ensemble_introduction.ipynb
│   ├── 2. ensemble_random_forest.ipynb
│   ├── 3. ensemble_bagging.ipynb
│   ├── 4. ensemble_gradient_boosting.ipynb
│   ├── 5. ensemble_adaboost.ipynb
│   ├── 6. ensemble_methods_all.ipynb (comprehensive reference)
│   └── datasets/
├── activities/
│   ├── Bag-and-Boost/
│   ├── Random_Forests/
│   ├── Encoding_Categoricals/
│   ├── Ins_Hyperparameters/
│   └── Stu_Hyperparameters/
└── exercises/
    ├── ensemble_ex_01.ipynb - ensemble_ex_04.ipynb
    └── solutions/
```

---

## 🔄 Recommended Learning Path

### **Part 1: Foundations** (2-3 hours)

1. **Introduction to Ensembles**: `overview/1. ensemble_introduction.ipynb`
2. **Random Forests**: `overview/2. ensemble_random_forest.ipynb`
3. **Bagging Techniques**: `overview/3. ensemble_bagging.ipynb`

### **Part 2: Boosting Methods** (2-3 hours)

4. **Gradient Boosting**: `overview/4. ensemble_gradient_boosting.ipynb`
5. **AdaBoost**: `overview/5. ensemble_adaboost.ipynb`

### **Part 3: Hands-on Practice** (2-3 hours)

6. **Activities**: Work through `activities/` folders
7. **Exercises**: Complete `exercises/ensemble_ex_01-04.ipynb`

---

## 🔍 Methods Covered

| Method | Type | Description | Best For |
|--------|------|-------------|----------|
| **Random Forest** | Bagging | Bootstrap aggregating with decision trees | High-variance models, feature importance |
| **Bagging** | Variance Reduction | Train on bootstrap samples, average predictions | Reducing overfitting |
| **Gradient Boosting** | Boosting | Sequential trees correcting errors | High accuracy, tabular data |
| **AdaBoost** | Boosting | Weight misclassified examples | Binary classification |
| **XGBoost/LightGBM** | Advanced Boosting | Optimized gradient boosting | Competition-level performance |

---

## 📊 Datasets Used

- **Adult Census Dataset**: Income classification (>50K or ≤50K)
- **Penguins Dataset**: Species classification and regression tasks

---

## 🛠️ Technical Requirements

```python
# Core
numpy, pandas, matplotlib, seaborn

# Machine Learning
scikit-learn, xgboost, lightgbm
```

---

## 🔗 Related Modules

- **Prerequisites**: [Linear Models](../linear_models/), [Cross Validation](../cross_validation/)
- **Next Steps**: [Explainability](../explainability/), [Hyperparameter Optimization](../optimization/)

---

*Module Difficulty: Intermediate*  
*Estimated Time: 6-9 hours total*
