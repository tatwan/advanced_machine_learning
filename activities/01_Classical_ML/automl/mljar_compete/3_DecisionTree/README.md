# Summary of 3_DecisionTree

[<< Go back](../README.md)


## Decision Tree
- **n_jobs**: -1
- **criterion**: gini
- **max_depth**: 2
- **explain_level**: 0

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.9
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
logloss

## Training time

0.6 seconds

## Metric details
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.385046 | nan         |
| auc       | 0.906694 | nan         |
| f1        | 0.896552 |   0.0551724 |
| accuracy  | 0.869565 |   0.0551724 |
| precision | 0.96     |   0.888889  |
| recall    | 1        |   0.0496552 |
| mcc       | 0.744965 |   0.888889  |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.385046 | nan         |
| auc       | 0.906694 | nan         |
| f1        | 0.896552 |   0.0551724 |
| accuracy  | 0.869565 |   0.0551724 |
| precision | 0.896552 |   0.0551724 |
| recall    | 0.896552 |   0.0551724 |
| mcc       | 0.720081 |   0.0551724 |


## Confusion matrix (at threshold=0.055172)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |               14 |                3 |
| Labeled as 1 |                3 |               26 |

## Learning curves
![Learning curves](learning_curves.png)
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



[<< Go back](../README.md)
