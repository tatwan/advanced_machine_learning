# Machine Learning Methods for Anomaly Detection

## 📚 Overview

This module focuses on **machine learning approaches** to anomaly detection, moving beyond traditional statistical methods to leverage powerful algorithms from the **PyOD (Python Outlier Detection)** library. You will learn how to apply various ML algorithms to both cross-sectional (non-time series) and time series data, understanding the nuances of preprocessing and model selection.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Apply **6+ different machine learning algorithms** (KNN, LOF, CBLOF, COPOD, ECOD, OCSVM)
- ✅ Understand the strengths and weaknesses of different algorithm families (Distance, Clustering, Probabilistic, Kernel)
- ✅ Detect anomalies in **time series data** using advanced preprocessing techniques
- ✅ Implement **sliding window** and **feature engineering** strategies for temporal context
- ✅ Evaluate model performance using ground truth data and standard metrics
- ✅ Compare how different preprocessing methods affect anomaly detection results

---

## 📂 Module Structure

This module contains **2 main activities** organized as follows:

```
anomaly_detection_ml/
├── README.md (this file)
├── activities/
│   ├── activity_01_ml_anomaly_basics.ipynb
│   └── activity_02_ml_anomaly_timeseries.ipynb
├── data/
│   ├── weight-height.csv
│   └── nyc_taxi.csv
├── demos/
│   └── (Demo notebooks if applicable)
└── solutions/
    ├── activity_01_ml_anomaly_basics_SOLUTION.ipynb
    └── activity_02_ml_anomaly_timeseries_SOLUTION.ipynb
```

### **Notebook Descriptions:**

- **activity_01_ml_anomaly_basics.ipynb**: Introduction to ML anomaly detection on non-time series data. Covers a wide range of algorithms.
- **activity_02_ml_anomaly_timeseries.ipynb**: Advanced application of ML algorithms to time series data, focusing on preprocessing strategies like decomposition and sliding windows.

---

## 🔄 Recommended Learning Path

### **Part 1: ML Anomaly Detection Basics** (2-3 hours)

#### Activity: `activities/activity_01_ml_anomaly_basics.ipynb`

**Focus:**
- Introduction to **PyOD** library
- Implementing algorithms from different families:
    - **Distance-Based**: KNN, LOF
    - **Clustering-Based**: CBLOF
    - **Probabilistic**: COPOD, ECOD
    - **Kernel-Based**: One-Class SVM (OCSVM)
- Understanding the impact of **feature scaling** (especially for OCSVM)
- Comparing algorithm performance on the **Weight-Height dataset**

### **Part 2: Time Series Anomaly Detection with ML** (2-3 hours)

#### Activity: `activities/activity_02_ml_anomaly_timeseries.ipynb`

**Focus:**
- Applying ML algorithms (KNN, LOF, Isolation Forest) to **time series data**
- **Preprocessing Strategies**:
    1.  **Point Outliers**: Raw values (ignoring time)
    2.  **Decomposition**: Residuals from STL decomposition
    3.  **Sliding Windows**: Capturing temporal patterns
    4.  **Feature Engineering**: Adding context (day of week, holidays)
- Detecting known anomalies in the **NYC Taxi dataset**
- Evaluating how preprocessing changes which anomalies are detected

---

## 🔍 Methods Covered

### **Activity 1: ML Algorithms (Non-Time Series)**

| Algorithm Family | Method | Description | Pros | Cons |
| :--- | :--- | :--- | :--- | :--- |
| **Distance-Based** | **KNN** (K-Nearest Neighbors) | Uses distance to k-th neighbor as outlier score | Simple, interpretable | Slow for large data |
| | **LOF** (Local Outlier Factor) | Compares local density to neighbors | Handles varying densities | Slower than KNN |
| **Clustering-Based** | **CBLOF** (Cluster-Based LOF) | Clusters data, scores based on distance to cluster | Handles complex distributions | Need to choose k clusters |
| **Probabilistic** | **COPOD** (Copula-Based) | Models tail probabilities using copulas | Parameter-free, fast | Assumes independence (mostly) |
| | **ECOD** (Empirical CDF) | Uses empirical CDF for tail probabilities | Very fast, simple | Assumes independence |
| **Kernel-Based** | **OCSVM** (One-Class SVM) | Learns boundary around normal data | Captures non-linearities | Sensitive to scaling, slow |

### **Activity 2: Time Series Preprocessing Strategies**

| Strategy | Description | Best For | What it Detects |
| :--- | :--- | :--- | :--- |
| **Point Outliers** | Apply ML to raw values | Simple magnitude checks | Extreme high/low values |
| **Decomposition** | Apply ML to STL residuals | Seasonal data | Deviations from seasonal pattern |
| **Sliding Windows** | Create windows of N days | Pattern recognition | Unusual sequences/shapes |
| **Feature Engineering** | Add time features (dow, holiday) | Context-aware detection | Contextual anomalies (e.g., high traffic on holiday) |

---

## 📊 Datasets Used

### **Weight-Height Dataset (Activity 1)**
- **Type**: Cross-sectional
- **Features**: Height, Weight
- **Purpose**: Demonstrate univariate and multivariate outlier detection across different algorithm families.

### **NYC Taxi Dataset (Activity 2)**
- **Type**: Time Series (Daily)
- **Features**: Passenger counts
- **Known Anomalies**: Holidays (Thanksgiving, Christmas), Events (Marathon), Weather (Blizzard).
- **Purpose**: Evaluate how different preprocessing techniques enable standard ML algorithms to detect temporal anomalies.

---

## 🛠️ Technical Requirements

### **Required Libraries:**
```python
# Core
numpy
pandas
matplotlib
seaborn

# Machine Learning & Anomaly Detection
pyod        # Python Outlier Detection
scikit-learn
statsmodels # For STL decomposition
holidays    # For feature engineering
```

### **Installation:**
```bash
pip install numpy pandas matplotlib seaborn pyod scikit-learn statsmodels holidays
```

---

## 🆘 Getting Help

- **PyOD Documentation**: [https://pyod.readthedocs.io/](https://pyod.readthedocs.io/)
- **Check Solution Notebooks**: If you get stuck, refer to the `solutions/` folder.
- **Common Issues**:
    - `ModuleNotFoundError`: Ensure you installed `pyod` and `holidays`.
    - **OCSVM Results**: If OCSVM performs poorly, check if you scaled your data!
