# Linear Models & Regularization

## 📚 Overview

This module provides comprehensive coverage of **linear and logistic regression**, along with **regularization techniques** (L1, L2, ElasticNet) for model improvement. You will learn foundational supervised learning techniques that form the basis for more advanced methods.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Implement **linear regression** with feature engineering
- ✅ Apply **logistic regression** for binary and multi-class classification
- ✅ Understand and apply **regularization** (Lasso, Ridge, ElasticNet)
- ✅ Perform **feature selection** using regularization techniques
- ✅ Build classification models (SVM, KNN) and compare performance
- ✅ Evaluate models using appropriate metrics

---

## 📂 Module Structure

```
linear_models/
├── README.md (this file)
├── overview/
│   ├── linear_models_feature_engineering_classification.ipynb
│   ├── linear_models_regularization.ipynb
│   ├── logistic_regression.ipynb
│   ├── datasets/
│   ├── exercises/
│   └── solutions/
├── activities/
│   └── Various activity folders
└── practice/
    ├── 01-Ins_Linear_Regression/ & 01-Stu_Predicting_Sales/
    ├── 02-Ins_Logistic_Regression/ & 02-Stu_Logistic_Regression/
    ├── 03-Ins_Regressions_and_Regularization/ & 03-Stu_Regularization/
    ├── 04-Ins_Classification_Models/ & 04-Stu_Classification_Models/
    └── 05-Ins_SVM/ & 05-Stu_SVM/
```

---

## 🔄 Recommended Learning Path

### **Part 1: Linear Regression** (2-3 hours)

1. **Demo**: `practice/01-Ins_Linear_Regression/`
2. **Practice**: `practice/01-Stu_Predicting_Sales/`
3. **Deep Dive**: `overview/linear_models_feature_engineering_classification.ipynb`

### **Part 2: Logistic Regression** (2 hours)

4. **Demo**: `practice/02-Ins_Logistic_Regression/`
5. **Practice**: `practice/02-Stu_Logistic_Regression/`
6. **Deep Dive**: `overview/logistic_regression.ipynb`

### **Part 3: Regularization** (2-3 hours)

7. **Demo**: `practice/03-Ins_Regressions_and_Regularization/`
8. **Practice**: `practice/03-Stu_Regularization/` & `practice/03-Stu_Lasso-Feature-Selection/`
9. **Deep Dive**: `overview/linear_models_regularization.ipynb`

### **Part 4: Classification & SVM** (2 hours)

10. **Classification Models**: `practice/04-Ins/Stu_Classification_Models/`
11. **Support Vector Machines**: `practice/05-Ins/Stu_SVM/`

---

## 🔍 Methods Covered

| Method | Type | Description |
|--------|------|-------------|
| **Linear Regression** | Regression | Predict continuous outcomes with linear relationships |
| **Logistic Regression** | Classification | Binary/multi-class classification with probability outputs |
| **Ridge (L2)** | Regularization | Shrinks coefficients, handles multicollinearity |
| **Lasso (L1)** | Regularization | Feature selection via coefficient zeroing |
| **ElasticNet** | Regularization | Combines L1 and L2 penalties |
| **SVM** | Classification | Maximum margin classifier with kernel options |

---

## 🛠️ Technical Requirements

```python
numpy, pandas, matplotlib, seaborn, scikit-learn
```

---

## 🔗 Related Modules

- **Next Steps**: [Ensemble Methods](../ensemble/), [Cross Validation](../cross_validation/)
- **Advanced**: [GAMs & GLMs](../gams_glms/)

---

*Module Difficulty: Beginner to Intermediate*  
*Estimated Time: 8-10 hours total*
