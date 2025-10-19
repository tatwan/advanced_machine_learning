# Linear Models Optimization and Evaluation - Quick Reference

## Notebook Contents Summary

### Total Cells: 37 (13 Markdown + 24 Code)

## Part 1: Optimizing and Training Linear Models

### 1.1 Understanding Gradient Descent Optimization
- **Concepts**: Batch, Stochastic, Mini-batch gradient descent
- **Implementation**: SGDClassifier with different learning rates
- **Comparison**: 4 learning rate schedules (constant, optimal, invscaling, adaptive)
- **Visualization**: Bar chart comparing performance

### 1.2 L1 and L2/ElasticNet Regularization
- **Algorithms**: No regularization, L1 (Lasso), L2 (Ridge), ElasticNet
- **Demonstration**: Feature selection with L1
- **Visualization**: Coefficient magnitudes for each regularization type
- **Key Insight**: L1 produces sparse models suitable for feature selection

### 1.3 Support Vector Machines
- **Kernels tested**: Linear, RBF, Polynomial, Sigmoid
- **Metrics**: Accuracy, Precision, Recall, F1-Score for each kernel
- **Visualization**: Performance comparison across kernels

### 1.4 k-Folds Cross Validation
- **k values**: 3, 5, 10 folds
- **Models compared**: Logistic Regression, SGD, Linear SVM, RBF SVM
- **Visualization**: Bar chart with error bars showing CV scores

## Part 2: Evaluating Models

### 2.1 Metrics for Model Evaluation
- **Basic metrics**: Accuracy, Precision, Recall, F1-Score
- **Advanced**: Classification report, Confusion matrix
- **Visualization**: Heatmap of confusion matrix with interpretation

### 2.2 Hyperparameter Tuning
- **Method**: GridSearchCV
- **Parameters**: C values (6 options), penalty types (L1, L2)
- **Total combinations**: 12 parameter sets
- **Visualization**: Heatmap showing F1-score by C and penalty

### 2.3 Threshold Tuning
- **Thresholds tested**: 0.3, 0.4, 0.5, 0.6, 0.7, 0.8
- **Analysis**: Precision-Recall trade-off
- **Visualizations**: 
  - Precision & Recall vs Threshold
  - Accuracy & F1-Score vs Threshold
- **Use case**: Medical diagnosis requiring high recall

### 2.4 Handling Class Imbalance
- **Strategies**:
  1. No handling (baseline)
  2. Class weights ('balanced')
  3. SMOTE (oversampling)
  4. Random undersampling
- **Comparison**: All 4 strategies on metrics
- **Visualizations**: 
  - F1-Score by strategy
  - Precision & Recall comparison

### 2.5 Advanced Metrics
- **ROC Curve**: True Positive Rate vs False Positive Rate
- **ROC-AUC**: Area under ROC curve
- **Precision-Recall Curve**: For imbalanced datasets
- **Average Precision**: PR curve summary
- **Visualizations**: Side-by-side ROC and PR curves

## Dataset: Breast Cancer Wisconsin

- **Source**: UCI ML Repository (via scikit-learn)
- **Samples**: 569
- **Features**: 30 numerical
- **Target**: Binary (Malignant=0, Benign=1)
- **Split**: 80% train, 20% test (stratified)
- **Preprocessing**: StandardScaler applied

## Key Code Examples

### Loading Data
```python
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X, y = data.data, data.target
```

### Gradient Descent
```python
sgd_clf = SGDClassifier(loss='log_loss', learning_rate='optimal')
```

### Regularization
```python
# L1
LogisticRegression(penalty='l1', C=1.0, solver='liblinear')
# L2
LogisticRegression(penalty='l2', C=1.0)
# ElasticNet
LogisticRegression(penalty='elasticnet', l1_ratio=0.5, solver='saga')
```

### SVM
```python
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
```

### Cross Validation
```python
scores = cross_val_score(model, X_train, y_train, cv=5)
```

### Grid Search
```python
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1')
```

### Threshold Tuning
```python
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= threshold).astype(int)
```

### Class Imbalance
```python
# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Class Weights
model = LogisticRegression(class_weight='balanced')
```

### ROC Curve
```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
```

## Visualizations Included

1. **Class Distribution**: Bar chart
2. **Learning Rate Comparison**: Bar chart
3. **Coefficient Magnitudes**: 2x2 subplot (4 regularization types)
4. **SVM Kernel Comparison**: 2 bar charts (Accuracy & F1)
5. **Cross-Validation**: Bar chart with error bars
6. **Confusion Matrix**: Heatmap with annotations
7. **Hyperparameter Heatmap**: GridSearch results
8. **Threshold Analysis**: 2 line plots
9. **Imbalance Strategies**: 2 bar charts
10. **ROC & PR Curves**: Side-by-side plots

## Learning Outcomes

After completing this notebook, you will be able to:

✓ Understand and implement gradient descent optimization  
✓ Apply regularization to prevent overfitting  
✓ Use SVMs with different kernels  
✓ Perform robust model evaluation with cross-validation  
✓ Calculate and interpret comprehensive metrics  
✓ Tune hyperparameters systematically  
✓ Adjust decision thresholds for business needs  
✓ Handle class imbalance effectively  
✓ Use advanced metrics like ROC-AUC and PR-AUC  

## Time to Complete

- **Quick run-through**: 30-45 minutes (just run all cells)
- **With reading**: 1.5-2 hours (read markdown, understand outputs)
- **Deep dive**: 3-4 hours (modify parameters, experiment)

## Prerequisites

- Python 3.7+
- Libraries: numpy, pandas, scikit-learn, matplotlib, seaborn, imbalanced-learn
- Basic ML knowledge
- Understanding of classification problems

## Next Steps

After this notebook, consider:
1. Try with different datasets (diabetes, iris, wine)
2. Implement custom metrics
3. Add more algorithms (Random Forest, XGBoost)
4. Deep dive into specific topics
5. Apply to your own datasets

---

**Created for**: Advanced Machine Learning Course  
**Author**: Generated for tatwan/advanced_machine_learning  
**Last Updated**: October 2025
