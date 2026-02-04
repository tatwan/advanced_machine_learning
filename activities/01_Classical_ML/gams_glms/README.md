# Generalized Linear Models (GLMs) & Generalized Additive Models (GAMs)

## 📚 Overview

This module covers **advanced linear modeling techniques** beyond standard linear regression. Learn to model non-linear relationships and non-normal distributions using GLMs and GAMs.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Understand when to use **GLMs** over standard linear regression
- ✅ Apply different **link functions** (logit, log, identity)
- ✅ Model **non-normal distributions** (Poisson, Binomial, Gamma)
- ✅ Use **GAMs** for flexible non-linear relationships
- ✅ Interpret **interaction terms** and **feature effects**
- ✅ Apply smoothing functions for continuous predictors

---

## 📂 Module Structure

```
gams_glms/
├── README.md (this file)
└── Advanced_Linear_Models.ipynb
```

---

## 🔄 Learning Path

### **Complete Lab** (3-4 hours)

Work through `Advanced_Linear_Models.ipynb` covering:

1. **GLM Fundamentals**: Link functions, exponential family distributions
2. **Poisson Regression**: Count data modeling
3. **Logistic Regression as GLM**: Binary outcomes
4. **GAM Introduction**: Splines and smooth functions
5. **Interaction Effects**: Modeling feature interactions
6. **Model Comparison**: GLM vs GAM selection

---

## 🔍 Models Covered

| Model | Distribution | Link Function | Use Case |
|-------|--------------|---------------|----------|
| **Linear GLM** | Normal | Identity | Standard regression |
| **Logistic GLM** | Binomial | Logit | Binary classification |
| **Poisson GLM** | Poisson | Log | Count data |
| **Gamma GLM** | Gamma | Log | Positive continuous |
| **GAM** | Various | Various | Non-linear relationships |

---

## 🛠️ Technical Requirements

```python
numpy, pandas, statsmodels, pygam, scikit-learn
```

---

## 🔗 Related Modules

- **Prerequisites**: [Linear Models](../linear_models/)
- **Related**: [Explainability](../explainability/)

---

*Module Difficulty: Advanced*  
*Estimated Time: 3-4 hours total*
