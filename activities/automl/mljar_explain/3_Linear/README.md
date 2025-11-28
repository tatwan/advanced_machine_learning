# Summary of 3_Linear

[<< Go back](../README.md)


## Logistic Regression (Linear)
- **n_jobs**: -1
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.75
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
logloss

## Training time

2.1 seconds

## Metric details
|           |    score |     threshold |
|:----------|---------:|--------------:|
| logloss   | 0.112444 | nan           |
| auc       | 0.994709 | nan           |
| f1        | 0.979021 |   0.640217    |
| accuracy  | 0.973684 |   0.640217    |
| precision | 1        |   0.744524    |
| recall    | 1        |   2.50319e-09 |
| mcc       | 0.943898 |   0.640217    |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.112444 |  nan        |
| auc       | 0.994709 |  nan        |
| f1        | 0.979021 |    0.640217 |
| accuracy  | 0.973684 |    0.640217 |
| precision | 0.985915 |    0.640217 |
| recall    | 0.972222 |    0.640217 |
| mcc       | 0.943898 |    0.640217 |


## Confusion matrix (at threshold=0.640217)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |               41 |                1 |
| Labeled as 1 |                2 |               70 |

## Learning curves
![Learning curves](learning_curves.png)

## Coefficients
| feature    |    Learner_1 |
|:-----------|-------------:|
| intercept  |  1.99655     |
| feature_12 |  0.688197    |
| feature_2  |  0.368747    |
| feature_1  |  0.309169    |
| feature_4  |  0.1662      |
| feature_20 | -0.000325827 |
| feature_16 | -0.0119186   |
| feature_15 | -0.0150287   |
| feature_18 | -0.0353431   |
| feature_17 | -0.0454815   |
| feature_19 | -0.0585091   |
| feature_10 | -0.0604671   |
| feature_30 | -0.161914    |
| feature_5  | -0.182978    |
| feature_9  | -0.276784    |
| feature_3  | -0.287265    |
| feature_25 | -0.33386     |
| feature_8  | -0.380486    |
| feature_6  | -0.429382    |
| feature_11 | -0.62289     |
| feature_28 | -0.687107    |
| feature_7  | -0.721931    |
| feature_29 | -0.892021    |
| feature_24 | -1.04878     |
| feature_21 | -1.22197     |
| feature_26 | -1.23803     |
| feature_14 | -1.23842     |
| feature_13 | -1.25887     |
| feature_23 | -1.27844     |
| feature_27 | -1.74648     |
| feature_22 | -1.9114      |


## Permutation-based Importance
![Permutation-based Importance](permutation_importance.png)
## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)


## Normalized Confusion Matrix

![Normalized Confusion Matrix](confusion_matrix_normalized.png)


## ROC Curve

![ROC Curve](roc_curve.png)


## Kolmogorov-Smirnov Statistic

![Kolmogorov-Smirnov Statistic](ks_statistic.png)


## Precision-Recall Curve

![Precision-Recall Curve](precision_recall_curve.png)


## Calibration Curve

![Calibration Curve](calibration_curve_curve.png)


## Cumulative Gains Curve

![Cumulative Gains Curve](cumulative_gains_curve.png)


## Lift Curve

![Lift Curve](lift_curve.png)



## SHAP Importance
![SHAP Importance](shap_importance.png)

## SHAP Dependence plots

### Dependence (Fold 1)
![SHAP Dependence from Fold 1](learner_fold_0_shap_dependence.png)

## SHAP Decision plots

### Top-10 Worst decisions for class 0 (Fold 1)
![SHAP worst decisions class 0 from Fold 1](learner_fold_0_shap_class_0_worst_decisions.png)
### Top-10 Best decisions for class 0 (Fold 1)
![SHAP best decisions class 0 from Fold 1](learner_fold_0_shap_class_0_best_decisions.png)
### Top-10 Worst decisions for class 1 (Fold 1)
![SHAP worst decisions class 1 from Fold 1](learner_fold_0_shap_class_1_worst_decisions.png)
### Top-10 Best decisions for class 1 (Fold 1)
![SHAP best decisions class 1 from Fold 1](learner_fold_0_shap_class_1_best_decisions.png)

[<< Go back](../README.md)
