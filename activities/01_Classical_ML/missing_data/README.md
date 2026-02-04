# Missing Data Handling

## 📚 Overview

This module covers **missing data diagnosis and imputation** techniques. Learn to assess data quality and apply appropriate strategies based on the missingness mechanism.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Diagnose **data quality issues** and missing patterns
- ✅ Apply **univariate imputation** (mean, median, mode)
- ✅ Implement **multivariate imputation** (KNN, Iterative Imputer)
- ✅ Use **time series interpolation** for temporal data
- ✅ Choose appropriate strategies based on missingness type (MCAR, MAR, MNAR)

---

## 📂 Module Structure

```
missing_data/
├── README.md (this file)
├── student_missing_data_v2.ipynb (Comprehensive lab)
├── data/
│   └── (Various datasets with missing values)
└── solution/
```

---

## 🔄 Learning Path

### **Complete Lab** (3-4 hours)

Work through `student_missing_data_v2.ipynb`:

1. **Data Quality Assessment**: Missing patterns, visualization
2. **Simple Imputation**: Mean, median, mode, constant
3. **Advanced Imputation**: KNN, Iterative Imputer
4. **Time Series**: Interpolation, forward/backward fill
5. **Comparison**: Impact on model performance

---

## 🔍 Methods Covered

| Method | Type | Best For |
|--------|------|----------|
| **Mean/Median** | Univariate | Numerical, MCAR |
| **Mode** | Univariate | Categorical |
| **KNN Imputer** | Multivariate | Similar observations exist |
| **Iterative Imputer** | Multivariate | Complex relationships |
| **Interpolation** | Time Series | Temporal continuity |

---

## 🛠️ Technical Requirements

```python
numpy, pandas, scikit-learn, missingno
```

---

## 🔗 Related Modules

- **Next**: [Preprocessing Pipelines](../preprocessing_pipeline/)

---

*Module Difficulty: Beginner to Intermediate*  
