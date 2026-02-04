### Machine Learning Metric Decision Tree (v5 - Complete)

## 1. **Is the task Supervised or Unsupervised Learning?**
   - **Supervised Learning** (You have labeled data) → Go to 2
   - **Unsupervised Learning** (You have unlabeled data, for clustering) → Go to 9

---

## 2. **What type of Supervised Learning task is it?**
   - **Classification** (Predicting discrete class labels) → Go to 3
   - **Regression** (Predicting continuous values) → Go to 6
   - **Ranking** (Predicting an ordered list) → Go to 8
   - **Object Detection** (Locating and classifying objects in images) → See section 2a

### 2a. **Object Detection Metrics**
   - **IoU (Intersection over Union):** Measures overlap between predicted and ground truth bounding boxes. Typically use IoU threshold of 0.5 or 0.75.
   - **mAP (mean Average Precision):** The gold standard for object detection. Averages precision across different IoU thresholds and object classes.
   - **AP@[IoU] (Average Precision at specific IoU):** Common variants include AP@0.5, AP@0.75, and AP@[0.5:0.95] (COCO metric).
   - **Precision-Recall curves:** Visualize performance at different confidence thresholds.

---

## 3. **Classification: Problem Structure**

### 3a. **Is your data heavily imbalanced?**
   - **Yes** (e.g., 1% positive class, fraud detection, rare disease)
     - **Important:** Accuracy is misleading for imbalanced data!
     - **Strongly consider:**
       - **MCC (Matthews Correlation Coefficient):** Best single metric for imbalanced binary classification. Ranges from -1 to +1. Accounts for all confusion matrix values.
       - **Cohen's Kappa:** Measures agreement beyond chance. Good for imbalanced datasets.
       - **AUC-PR (Precision-Recall AUC):** Better than AUC-ROC for imbalanced data.
       - **F1-Score or F-beta:** Balances precision and recall.
     - Then proceed to 3b for detailed guidance.
   - **No** (Relatively balanced classes) → Go to 3b

### 3b. **How many classes are you predicting?**
   - **Binary Classification** (2 classes: positive/negative) → Go to 4
   - **Multi-Class Classification** (3+ mutually exclusive classes) → Go to 5
   - **Multi-Label Classification** (Each instance can have multiple labels) → See section 5a

---

## 4. **Binary Classification: Business Goal & Error Cost**

### **What is more costly: a False Positive or a False Negative?**

- **False Negatives are more costly** (e.g., missing a disease diagnosis, failing to detect fraud)
  - You want to find as many true positives as possible.
  - **Primary Metric: Recall (Sensitivity, True Positive Rate)**
  - **Also consider: F-beta Score (with β > 1)** to favor recall (e.g., F2-Score with β=2)

- **False Positives are more costly** (e.g., flagging legitimate email as spam, false alarms)
  - You want your positive predictions to be correct.
  - **Primary Metric: Precision (Positive Predictive Value)**
  - **Also consider: F-beta Score (with β < 1)** to favor precision (e.g., F0.5-Score with β=0.5)

- **Both are equally costly, or you need a balanced measure**
  - **Primary Metric: F1-Score** (harmonic mean of Precision and Recall)
  - **Alternative: MCC (Matthews Correlation Coefficient)** - Often better than F1 for imbalanced data. Single value from -1 to +1.

- **Correctly identifying true negatives is critical** (e.g., a "rule-out" test for a disease; you must be sure when something is negative)
  - **Primary Metric: Specificity (True Negative Rate)**
  - **Also consider: NPV (Negative Predictive Value)**

### **Next, consider how you will use the model's output** → Go to 4a

### 4a. **Binary Classification: Output Type & Thresholds**

- **Do you need the quality of the predicted probabilities themselves?** (e.g., for risk scoring, financial modeling, decision-making under uncertainty)
  - **Yes → Use Probabilistic Scoring Rules:**
    - **Log-Loss (Binary Cross-Entropy):** Penalizes models heavily for being confident and wrong. Lower is better.
    - **Brier Score:** Mean squared error between predicted probabilities and actual outcomes (0 or 1). Lower is better.
    - **Calibration Metrics:**
      - **ECE (Expected Calibration Error):** Measures how well predicted probabilities match actual frequencies. Lower is better.
      - **Reliability diagrams:** Visualize calibration quality.

- **Do you need to evaluate performance across all possible decision thresholds?** (i.e., comparing models independent of a specific threshold)
  - **Yes → Use Threshold-Sweeping Metrics:**
    - **AUC-ROC (Area Under the ROC Curve):** Standard for measuring a model's ability to discriminate between classes. Good for balanced datasets.
    - **AUC-PR (Area Under the Precision-Recall Curve):** **Strongly preferred over AUC-ROC for highly imbalanced datasets.** Focuses on the positive class and is more informative for rare-event problems.

- **Do you need to evaluate performance at a single, fixed decision threshold?**
  - **Yes → Use Confusion Matrix-Based Metrics:**
    - **Accuracy:** Good for **balanced datasets only**. Misleading for imbalanced data.
    - **Precision, Recall, F1-Score, Specificity:** Choose based on the error cost analysis above.
    - **MCC (Matthews Correlation Coefficient):** Single comprehensive metric. Excellent for imbalanced data.
    - **Cohen's Kappa:** Measures agreement beyond chance.
    - **Always inspect the Confusion Matrix:** Understand the specific types of errors (TP, FP, TN, FN).

---

## 5. **Multi-Class Classification**

### 5a. **Is the problem multi-label? (Can instances have multiple classes simultaneously?)**
   - **Yes → Multi-Label Classification Metrics:**
     - **Hamming Loss:** Fraction of wrong labels (lower is better)
     - **Jaccard Score (Jaccard Index):** Intersection over union of predicted and true labels
     - **F1-Score (Micro/Macro/Weighted/Samples):**
       - **Micro:** Calculate globally across all labels
       - **Macro:** Calculate per label, then average (treats all labels equally)
       - **Weighted:** Weighted average by label frequency
       - **Samples:** Calculate per instance, then average
     - **Subset Accuracy:** Fraction of instances where all labels are correctly predicted (very strict)
     - **Label Ranking Average Precision:** For ranked multi-label predictions
   
   - **No → Continue to 5b**

### 5b. **Multi-Class Metrics (Mutually Exclusive Classes)**

- **For overall performance:**
  - **Accuracy:** Good when classes are balanced
  - **Balanced Accuracy:** Average of recall for each class. Better for imbalanced datasets.
  - **Cohen's Kappa:** Measures agreement beyond chance. Good for imbalanced multi-class problems.
  - **MCC (Matthews Correlation Coefficient):** Can be extended to multi-class. Comprehensive metric.

- **For per-class performance:**
  - **Precision, Recall, F1-Score per class:** Use macro, micro, or weighted averaging
    - **Macro-average:** Treats all classes equally (unweighted mean)
    - **Micro-average:** Weights by support (total predictions per class)
    - **Weighted-average:** Weights by true class frequencies
  
- **Do you care if the true class is in the top-K predictions?** (e.g., ImageNet uses top-5 accuracy)
  - **Yes → Use Top-K Accuracy:** Model is correct if true label is in top K predictions
  - Common for problems with many classes (e.g., K=1, 3, 5)

- **For probability quality:**
  - **Categorical Cross-Entropy (Log-Loss):** Multi-class extension of binary log-loss
  - **Expected Calibration Error (ECE):** Measures probability calibration

- **For visual analysis:**
  - **Confusion Matrix:** See which classes are confused with each other
  - **Classification Report:** Per-class precision, recall, F1-score

---

## 6. **Regression: Is this a time-series forecasting problem?**
   - **Yes** → Go to 7 (Time Series–Specific Metrics)
   - **No** → Proceed with general regression metrics below

### **General Regression Metrics:**

- **Are you comparing models across different scales or datasets?**
  - **Yes → Use Scale-Free or Relative Metrics:**
    - **R-squared (R²):** Proportion of variance explained. Range: (-∞, 1], where 1 is perfect.
    - **Adjusted R²:** Penalizes adding more features. Better for model comparison.
    - **MAPE (Mean Absolute Percentage Error):** Percentage-based error. **Warning:** Undefined when actual values are zero. Biased toward under-predictions.
    - **sMAPE (Symmetric MAPE):** More stable than MAPE near zero, but still has issues.
  
- **Are you working on the same scale?**
  - **Do you want to penalize large errors more heavily?**
    - **RMSE (Root Mean Squared Error):** Sensitive to outliers. Same units as target variable. Popular choice.
    - **MSE (Mean Squared Error):** Squared units. More sensitive to outliers than RMSE.
  
  - **Do you want to treat all errors equally (more robust to outliers)?**
    - **MAE (Mean Absolute Error):** Robust to outliers. Same units as target variable.
    - **Median Absolute Error:** Even more robust to outliers than MAE.

- **Do you need quantile predictions or uncertainty estimates?**
  - **Pinball Loss (Quantile Loss):** For quantile regression (e.g., 5th, 50th, 95th percentiles)
  - **Prediction Interval Coverage:** Percentage of actuals falling within prediction intervals

- **For financial or business metrics:**
  - **MASE (Mean Absolute Scaled Error):** See time series section
  - **Custom business metrics:** Revenue impact, cost savings, etc.

---

## 7. **Time Series–Specific Metrics & Notes**

Use the same core error metrics as regression (MAE, RMSE), but with special considerations.

### **Core Metrics:**
- **MAE (Mean Absolute Error):** Standard baseline. Robust to outliers.
- **RMSE (Root Mean Squared Error):** Penalizes large errors more. Common in competitions.

### **Comparing across different series or scales:**
- **Yes → Use Scale-Free Metrics:**
  - **MASE (Mean Absolute Scaled Error):** Divides MAE by the MAE of a naive forecast (usually seasonal naive or random walk). Values < 1 mean your model beats the naive baseline.
  - **RMSSE (Root Mean Squared Scaled Error):** RMSE version of MASE.
  - These are the **gold standard** for comparing forecasts across different time series with different scales.

### **Percentage Errors (use with caution):**
- **MAPE (Mean Absolute Percentage Error):** 
  - **Warnings:** Undefined for zero actuals, unstable near zero, biased toward under-predictions
  - **Avoid for:** Count data, intermittent demand, any data with zeros
- **sMAPE (Symmetric MAPE):** More stable than MAPE, but still has issues with zeros and asymmetry

### **Directional Accuracy:**
- **DA (Directional Accuracy):** Percentage of times the forecast correctly predicts the direction of change
- Useful when direction matters more than magnitude (e.g., stock movement up/down)

### **Important Notes:**
- **Temporal Cross-Validation is crucial:** Use rolling window, expanding window, or walk-forward validation
- **Never randomly shuffle time series data** for train/test splits
- Consider **forecast horizon:** Different metrics may be appropriate for short-term vs long-term forecasts
- **Seasonality matters:** Use seasonal naive baselines for MASE calculation when appropriate

---

## 8. **Ranking / Recommendation Metrics**

*Guideline: If users see only a short top-K list (e.g., 5-10 items), prioritize metrics at that K, such as Precision@K and NDCG@K.*

### **Do you care most about the order and position of relevant items in the list?**
- **Yes → Use Rank-Sensitive Metrics:**
  - **NDCG@K (Normalized Discounted Cumulative Gain):** The industry standard. Rewards putting *highly relevant* items at the very top. Handles graded relevance (e.g., 0-5 star ratings).
  - **MAP@K (Mean Average Precision):** Averages precision across the positions of all relevant items. Binary relevance only.
  - **MRR@K (Mean Reciprocal Rank):** Focuses only on how high the *first* relevant item is ranked. Good for "find one good answer" scenarios.
  - **ERR (Expected Reciprocal Rank):** Cascade model assuming users stop when finding a relevant item.

### **Are you mainly evaluating if relevant items appear in the top-K results, regardless of order?**
- **Yes → Use Set-Based Metrics:**
  - **Precision@K:** What fraction of your top-K recommendations are relevant?
  - **Recall@K:** What fraction of *all possible* relevant items did you find in your top-K list?
  - **F1-Score@K:** Harmonic mean of Precision@K and Recall@K.
  - **Hit Rate@K (HR@K):** Did at least one relevant item appear in top-K? (Binary: yes/no)

### **For personalized recommendations:**
- **Coverage:** Percentage of catalog items that get recommended to at least one user
- **Diversity:** How different are the recommended items from each other?
- **Novelty:** Are you recommending popular items or helping users discover new things?
- **Serendipity:** Surprising yet relevant recommendations

---

## 9. **Unsupervised Learning (Clustering) Metrics**

### **Do you have ground truth labels to compare against?**

- **Yes → Use External Validation Metrics:**
  - **Adjusted Rand Index (ARI):** Measures similarity between true and predicted clusters, correcting for chance. Range: [-1, 1], where 1 is perfect agreement.
  - **Adjusted Mutual Information (AMI):** Measures mutual information between true and predicted clusters, adjusted for chance.
  - **Homogeneity:** Are all clusters members of a single class? (0 to 1)
  - **Completeness:** Are all members of a class in the same cluster? (0 to 1)
  - **V-measure:** Harmonic mean of homogeneity and completeness.
  - **Fowlkes-Mallows Index:** Geometric mean of precision and recall at the pair level.

- **No ground truth labels? (Most common case)**
  - **No → Use Internal Validation Metrics:**
    - **Silhouette Coefficient:** Measures how similar an object is to its own cluster versus others. Range: [-1, 1]. Higher is better.
      - Values near 1: Well-clustered
      - Values near 0: On decision boundary
      - Values near -1: Possibly in wrong cluster
    - **Davies-Bouldin Index (DBI):** Measures average similarity between clusters. **Lower is better.** 0 is best.
    - **Calinski-Harabasz Index (CHI, Variance Ratio Criterion):** Ratio of between-cluster to within-cluster dispersion. **Higher is better.**
    - **Dunn Index:** Ratio of minimum inter-cluster distance to maximum intra-cluster distance. Higher is better, but computationally expensive.
    
    *Note: These metrics can disagree. It is wise to evaluate multiple metrics and use domain knowledge.*

### **How do you choose the number of clusters (k)?**
- **Elbow Method:** Plot metric (e.g., within-cluster sum of squares) vs. k; look for "elbow"
- **Silhouette Analysis:** Plot silhouette scores for different k values
- **Gap Statistic:** Compare within-cluster dispersion to null reference distribution
- **BIC/AIC:** For model-based clustering (e.g., Gaussian Mixture Models)
- **Domain knowledge:** Sometimes k is known from business context

---

## 10. **Special Considerations & Best Practices**

### **Cross-Validation Strategy:**
- **Standard ML:** k-fold cross-validation, stratified for classification
- **Time Series:** Walk-forward, expanding window, or rolling window validation
- **Small datasets:** Leave-one-out cross-validation (LOOCV)
- **Imbalanced data:** Stratified k-fold to maintain class proportions

### **Multiple Metrics:**
- **Always use multiple metrics** to get a complete picture
- Don't optimize for a single metric blindly
- Consider business impact alongside statistical metrics

### **Baseline Comparisons:**
- Always compare to simple baselines:
  - **Classification:** Majority class predictor, random predictor
  - **Regression:** Mean predictor, median predictor
  - **Time Series:** Naive forecast (last value), seasonal naive, moving average
  - **Ranking:** Random ranking, popularity-based ranking

### **Statistical Significance:**
- Use confidence intervals or statistical tests to determine if differences are meaningful
- Bootstrap resampling for robust confidence intervals
- Permutation tests for hypothesis testing

### **Fairness & Bias Metrics:**
- Consider fairness metrics if your model affects different demographic groups:
  - Demographic parity
  - Equal opportunity
  - Equalized odds
  - Disparate impact

---

## Quick Reference Summary

| **Task Type** | **Primary Metrics** | **Key Considerations** |
|---------------|---------------------|------------------------|
| **Binary Classification (Balanced)** | Accuracy, F1-Score, AUC-ROC | Balance precision/recall based on costs |
| **Binary Classification (Imbalanced)** | MCC, AUC-PR, F1-Score | Avoid accuracy; focus on minority class |
| **Multi-Class** | Balanced Accuracy, Macro F1, Cohen's Kappa | Consider per-class performance |
| **Multi-Label** | Hamming Loss, Jaccard Score, Micro/Macro F1 | Multiple averaging strategies |
| **Regression** | RMSE, MAE, R² | Choose based on outlier sensitivity |
| **Time Series** | MASE, RMSSE, MAE | Use scale-free metrics for comparisons |
| **Ranking** | NDCG@K, MAP@K, Precision@K | Focus on top-K performance |
| **Clustering (with labels)** | ARI, AMI, V-measure | External validation |
| **Clustering (no labels)** | Silhouette, DBI, CHI | Use multiple metrics |
| **Object Detection** | mAP, IoU, AP@[IoU] | Standard is COCO metrics |

---

**Version 5 - Complete Edition**  
*Comprehensive guide to choosing the right ML evaluation metric*
