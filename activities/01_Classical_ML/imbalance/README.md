# Evaluation Metrics for Classification: Balanced vs. Imbalanced Data

This guide provides an overview of evaluation metrics used in classification problems, highlighting the critical differences between handling balanced and imbalanced datasets. It serves as a companion to the `full_imbalanced_data_handling.ipynb` notebook.

## 1. Metrics for Balanced Data

When the classes in your dataset are roughly equal (e.g., 50% positive, 50% negative), standard metrics are usually sufficient.

### Accuracy
- **Definition**: The ratio of correctly predicted observations to the total observations.
- **Formula**: `(TP + TN) / (TP + TN + FP + FN)`
- **Usage**: Excellent for balanced datasets where all classes are equally important.

### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- **Definition**: Measures the ability of a classifier to distinguish between classes. Plots True Positive Rate (Sensitivity) vs. False Positive Rate (1-Specificity).
- **Usage**: Good for comparing models overall, but can be optimistic when the negative class is very large (imbalanced).

---

## 2. Metrics for Imbalanced Data

When one class significantly outnumbers the other (e.g., 95% negative, 5% positive), standard metrics like Accuracy can be misleading. A model that predicts the majority class for *every* instance can still achieve 95% accuracy but is useless for detecting the minority class.

### Why Accuracy Fails
In an imbalanced dataset (e.g., Fraud Detection), accuracy is biased towards the majority class. It hides the model's inability to correctly identify the rare, important events (the minority class).

### Preferred Metrics

#### Precision
- **Definition**: Of all instances predicted as positive, how many are actually positive?
- **Formula**: `TP / (TP + FP)`
- **Focus**: Minimizing False Positives. Important when the cost of a false alarm is high (e.g., spam filtering).

#### Recall (Sensitivity)
- **Definition**: Of all actual positive instances, how many did we correctly identify?
- **Formula**: `TP / (TP + FN)`
- **Focus**: Minimizing False Negatives. Critical when missing a positive case is dangerous (e.g., cancer diagnosis, fraud detection).

#### F1-Score
- **Definition**: The harmonic mean of Precision and Recall.
- **Formula**: `2 * (Precision * Recall) / (Precision + Recall)`
- **Usage**: Provides a single score that balances both concerns. Useful when you need a balance between Precision and Recall and there is an uneven class distribution.

#### PR-AUC (Precision-Recall Area Under Curve)
- **Definition**: The area under the Precision-Recall curve.
- **Usage**: **Superior to ROC-AUC for imbalanced data.** It focuses solely on the performance of the positive (minority) class and is not diluted by a large number of True Negatives.

#### Specificity
- **Definition**: The True Negative Rate.
- **Formula**: `TN / (TN + FP)`
- **Usage**: Measures how well the model identifies the majority class.

#### Geometric Mean (G-Mean)
- **Definition**: The geometric mean of Sensitivity (Recall) and Specificity.
- **Formula**: `sqrt(Sensitivity * Specificity)`
- **Usage**: Measures the balance between correctly classifying the majority and minority classes.

#### Index of Balanced Accuracy (IBA)
- **Definition**: A metric that quantifies the trade-off between the dominant and minority classes.
- **Usage**: Penalizes models that perform well on one class but poorly on the other.

---

## 3. Summary of `full_imbalanced_data_handling.ipynb`

The accompanying notebook provides a hands-on guide to applying these concepts and techniques.

### Key Topics Covered:
1.  **Data Generation**: Creating synthetic imbalanced datasets to simulate real-world scenarios.
2.  **Baseline Evaluation**: Demonstrating how standard Logistic Regression fails on imbalanced data using the metrics above.
3.  **Oversampling Techniques**:
    -   **Random Oversampling**: Duplicating minority samples.
    -   **SMOTE (Synthetic Minority Over-sampling Technique)**: Generating synthetic minority samples.
    -   **ADASYN**: Adaptive synthetic sampling.
4.  **Undersampling Techniques**:
    -   **Random Undersampling**: Removing majority samples.
    -   **Tomek Links**: Removing ambiguous samples near the decision boundary.
5.  **Combined Methods**: Using SMOTEENN and SMOTETomek to clean up overlapping data.
6.  **Ensemble Methods**: Using Balanced Random Forest and Easy Ensemble classifiers for robust performance.

### Key Takeaway
For imbalanced classification, **never rely on accuracy alone**. Always examine the Confusion Matrix, check Precision and Recall, and prioritize **PR-AUC** over ROC-AUC to get a true picture of your model's performance on the class that matters most.
