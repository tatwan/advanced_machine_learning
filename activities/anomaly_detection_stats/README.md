# Statistical Methods for Anomaly Detection

## 📚 Overview

This module provides comprehensive coverage of **statistical and mathematical approaches** to anomaly detection, covering both traditional univariate methods and advanced time-series-specific techniques. You will learn when and how to apply different statistical methods based on data characteristics, assumptions, and detection requirements.

---

## 🎯 Learning Objectives

By completing this module, you will be able to:

- ✅ Apply foundational statistical outlier detection methods (IQR, Z-scores, Modified Z-scores)
- ✅ Understand and implement multivariate outlier detection (Mahalanobis Distance, Isolation Forest)
- ✅ Detect anomalies in time series data using temporal context methods
- ✅ Choose appropriate methods based on data characteristics and assumptions
- ✅ Tune detection parameters and evaluate trade-offs between sensitivity and specificity
- ✅ Understand the difference between point anomalies, contextual anomalies, and pattern anomalies

---

## 📂 Module Structure

This module contains **6 notebooks** organized into three folders:

```
anomaly_detection_stats/
├── README.md (this file)
├── demos/
│   ├── 01_time_series_anomaly_detection.ipynb
│   └── 02_statistical_outlier_detection.ipynb
├── activities/
│   ├── activity_01_foundational_outlier_detection.ipynb
│   └── activity_02_time_series_anomaly_detection.ipynb
└── solutions/
    ├── activity_01_foundational_outlier_detection_SOLUTION.ipynb
    └── activity_02_time_series_anomaly_detection_SOLUTION.ipynb
```

### **Folder Descriptions:**

- **demos/** - Complete working examples of all methods with detailed explanations
- **activities/** - Hands-on exercises with TODO sections for you to complete
- **solutions/** - Completed versions of activities (for reference after attempting exercises)

---

## 🔄 Recommended Learning Path

### **Two-Part Learning Structure**

This module is divided into two main sections that build upon each other:

---

### **Part 1: Foundational Statistical Methods** (3-4 hours)

#### Step 1: Review the Demo Notebook (60-90 minutes)
📖 **Notebook:** `demos/02_statistical_outlier_detection.ipynb`

**What You'll Learn:**
- Visual outlier detection techniques (histograms, box plots, violin plots)
- **Tukey's IQR Method** - Quartile-based boundary detection
- **Z-Score Method** - Distribution-based detection with normality assumptions
- **Modified Z-Score** - Robust detection using median absolute deviation (MAD)
- When each method is appropriate based on data distribution

**What to Focus On:**
- Understand the mathematical concepts behind each method
- Note the assumptions each method makes (e.g., Z-score requires normality)
- Observe what happens when assumptions are violated
- Consider how these methods apply to real-world scenarios

#### Step 2: Complete the Activity (90-120 minutes)
✏️ **Notebook:** `activities/activity_01_foundational_outlier_detection.ipynb`

**Your Tasks:**
- Implement IQR, Z-score, and Modified Z-score detection functions
- **NEW:** Implement multivariate methods (Mahalanobis Distance, Isolation Forest)
- Complete parameter tuning exercises
- Analyze why multivariate methods are necessary
- Challenge exercise: Apply methods to new dataset and calculate precision/recall/F1

**What You'll Achieve:**
- Hands-on implementation of core statistical methods
- Understanding of when univariate methods fail
- Experience with multivariate outlier detection
- Performance evaluation using standard metrics

---

### **Part 2: Time Series Anomaly Detection** (3-4 hours)

#### Step 1: Review the Demo Notebook (60-90 minutes)
📖 **Notebook:** `demos/01_time_series_anomaly_detection.ipynb`

**What You'll Learn:**
- Time series data preparation and resampling strategies
- Visual exploration techniques (lag plots, autocorrelation)
- **Hampel Filter** - Rolling window-based detection with local median/MAD
- **STRAY** (Search TRace AnomalY) - Extreme value theory with concept drift handling
- **Matrix Profile** - Pattern-based subsequence anomaly detection
- Comparison of methods on NYC Taxi dataset with known anomalies

**What to Focus On:**
- Why time series data requires different approaches than independent data
- Concept of rolling windows and local vs. global context
- How concept drift affects anomaly detection
- Difference between point anomalies and pattern anomalies
- Note: This builds on Part 1 methods (IQR, Z-score), so complete Part 1 first

#### Step 2: Complete the Activity (90-120 minutes)
✏️ **Notebook:** `activities/activity_02_time_series_anomaly_detection.ipynb`

**Your Tasks:**
- Implement data resampling and aggregation strategies
- Create lag plots and interpret temporal dependencies
- Implement Hampel Filter core logic
- Experiment with STRAY parameters for concept drift detection
- Tune Matrix Profile for pattern anomaly detection
- Create comprehensive method comparison table
- Build decision guide for method selection

**What You'll Achieve:**
- Understanding of temporal context in anomaly detection
- Experience with rolling window methods
- Ability to tune complex algorithms (STRAY, Matrix Profile)
- Method selection skills based on anomaly type and data characteristics

---

## 🔍 Methods Covered

### **Foundational Univariate Methods**

| Method | Best For | Assumptions | Strengths | Limitations |
|--------|----------|-------------|-----------|-------------|
| **Tukey's IQR** | Any distribution | Distribution-free | Simple, robust to outliers in calculation | May miss outliers in skewed distributions |
| **Z-Score** | Normal distributions | Normality | Well-understood, interpretable | Sensitive to outliers, assumes Gaussian |
| **Modified Z-Score** | Non-normal data | None (uses median/MAD) | Robust to non-normality | Requires more computation |

### **Multivariate Methods**

| Method | Best For | Assumptions | Strengths | Limitations |
|--------|----------|-------------|-----------|-------------|
| **Mahalanobis Distance** | Correlated features | Linear relationships, covariance structure | Accounts for correlations | Requires sufficient samples for covariance estimation |
| **Isolation Forest** | High-dimensional data | None (tree-based) | Handles complex relationships, scalable | Black-box, harder to interpret |

### **Time Series-Specific Methods**

| Method | Best For | Assumptions | Strengths | Limitations |
|--------|----------|-------------|-----------|-------------|
| **Hampel Filter** | Local point anomalies | Stationarity in local windows | Simple, interpretable, online capable | Requires window size tuning, edge effects |
| **STRAY** | Concept drift, distribution shifts | None (data-driven) | Detects rare patterns, handles drift | Computationally intensive, parameter sensitivity |
| **Matrix Profile** | Pattern/subsequence anomalies | None (distance-based) | Finds repeated patterns, scale-invariant | Requires subsequence length tuning, memory intensive |

---

## 📊 Datasets Used

### **Activity 1: Weight-Height Dataset**
- **Type:** Cross-sectional, non-temporal
- **Features:** Height (inches), Weight (pounds)
- **Purpose:** Demonstrates univariate and multivariate detection differences
- **Characteristics:** Bivariate normal distribution with correlation

### **Activity 2: NYC Taxi Dataset (Numenta Anomaly Benchmark)**
- **Type:** Time series (30-minute intervals)
- **Period:** July 1, 2014 - May 31, 2015 (10,320 records)
- **Features:** Taxi passenger counts
- **Known Anomalies:**
  - NYC Marathon (Nov 1, 2014)
  - Thanksgiving (Nov 27, 2014)
  - Christmas (Dec 25, 2014)
  - New Year's Day (Jan 1, 2015)
  - North American Blizzard (Jan 27, 2015)
- **Purpose:** Real-world time series with ground truth for method evaluation

---

## 💡 Study Tips

### **Before Starting Activities:**
- Review the corresponding demo notebook thoroughly
- Understand the conceptual explanations before coding
- Note each method's assumptions and when they apply
- Keep the demo notebook open for reference

### **During Activities:**
- Read all instructions before starting each TODO
- Test your implementations incrementally
- Don't just aim for "working code" - understand WHY it works
- Experiment with parameters beyond the suggested values
- Use the solution notebooks only after attempting the exercises

### **After Completing Activities:**
- Review your method comparison tables
- Revisit reflection questions after completing all exercises
- Consider edge cases: What would break your implementation?
- Compare your solutions with the provided solution notebooks
- Think about how these methods apply to your domain (finance, IoT, healthcare, etc.)

---

## 🔗 Prerequisites and Connections

### **Prerequisites:**
- Basic statistics (mean, median, standard deviation, quartiles)
- Python programming (NumPy, Pandas)
- Data visualization (Matplotlib, Seaborn)
- Understanding of normal distribution

### **Related Topics:**
- **Machine Learning-Based Anomaly Detection:** Autoencoders, One-Class SVM, Local Outlier Factor (LOF)
- **Streaming Anomaly Detection:** Online algorithms, ADWIN, incremental methods
- **Multivariate Time Series:** VAR models, multivariate LSTM autoencoders
- **Evaluation Metrics:** Precision, Recall, F1-score, ROC-AUC for imbalanced detection
- **Concept Drift Detection:** Statistical tests, sliding windows, ADWIN
- **Explainability:** Interpreting why an observation was flagged as anomalous

### **Next Steps:**
- Apply methods to domain-specific datasets (finance, IoT, healthcare)
- Combine statistical and ML methods in ensemble approaches
- Implement streaming/online versions of these methods
- Explore deep learning approaches (LSTM autoencoders, Transformers)

---

## 🛠️ Technical Requirements

### **Required Libraries:**
```python
# Core
numpy
pandas
matplotlib
seaborn

# Statistical Methods
scipy
scikit-learn

# Time Series Specific
stumpy              # For Matrix Profile
stray               # For STRAY algorithm
```

### **Installation:**
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn stumpy stray
```

Or if a requirements file is provided:
```bash
pip install -r requirements.txt
```


## 🆘 Getting Help

If you encounter issues:

1. **Check demo notebooks** for working implementations and explanations
2. **Review error messages carefully** - they often point to the exact issue
3. **Verify data loading** - ensure datasets are in the correct location
4. **Check library versions** - run `pip list` to verify versions match requirements
5. **Solution notebooks** - Available in `solutions/` folder for reference
6. **Ask questions** - Use available support channels, discussion forums, or peer study groups

### **Common Issues:**

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'stray'` | Run `pip install stray` |
| Matrix Profile runs slowly | Use smaller subsequence length or subsample data |
| Z-score detects too many outliers | Check if data is normally distributed; consider Modified Z-score |
| Hampel filter misses anomalies | Increase window size or decrease n_sigma threshold |

---

## 🏆 Module Summary

This module provides a **comprehensive, hands-on introduction** to statistical anomaly detection methods. By completing both the demo reviews and hands-on activities, you will gain both theoretical understanding and practical skills in:

- **Foundational univariate methods** that apply to any distribution
- **Multivariate methods** that account for feature correlations
- **Time series-specific methods** that leverage temporal context
- **Method selection** based on data characteristics and detection goals
- **Performance evaluation** using standard metrics

### **Key Takeaways:**

Upon completion, you will be equipped to:
- ✅ Choose appropriate anomaly detection methods for different scenarios
- ✅ Implement and tune statistical detection algorithms
- ✅ Evaluate detection performance with appropriate metrics
- ✅ Understand trade-offs between different approaches
- ✅ Apply these methods to real-world problems in your domain
- ✅ Transition to more advanced ML-based detection methods

### **Application Domains:**

These methods are widely used in:
- **Finance:** Fraud detection, trading anomalies, risk management
- **Manufacturing:** Quality control, predictive maintenance, process monitoring
- **Healthcare:** Patient monitoring, clinical trial anomalies, disease outbreak detection
- **IT/Security:** Intrusion detection, system monitoring, performance anomalies
- **IoT:** Sensor monitoring, predictive maintenance, environmental anomalies
- **E-commerce:** Fraud detection, user behavior analysis, inventory anomalies

---


---

**Next Module:** Machine Learning-Based Anomaly Detection (Autoencoders, LOF, One-Class SVM)

---

*Module Difficulty: Intermediate*
*Estimated Time: 6-8 hours total*
*Last Updated: December 2025*
