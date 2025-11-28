# Summary of 19_CatBoost

[<< Go back](../README.md)


## CatBoost
- **n_jobs**: -1
- **learning_rate**: 0.1
- **depth**: 6
- **rsm**: 0.8
- **loss_function**: Logloss
- **eval_metric**: Logloss
- **explain_level**: 1

## Validation
 - **validation_type**: kfold
 - **k_folds**: 5
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
logloss

## Training time

3.3 seconds

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0941944 | nan           |
| auc       | 0.992138  | nan           |
| f1        | 0.97931   |   0.533114    |
| accuracy  | 0.973626  |   0.533114    |
| precision | 1         |   0.995304    |
| recall    | 1         |   9.06515e-05 |
| mcc       | 0.943648  |   0.533114    |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0941944 |  nan        |
| auc       | 0.992138  |  nan        |
| f1        | 0.97931   |    0.533114 |
| accuracy  | 0.973626  |    0.533114 |
| precision | 0.965986  |    0.533114 |
| recall    | 0.993007  |    0.533114 |
| mcc       | 0.943648  |    0.533114 |


## Confusion matrix (at threshold=0.533114)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |              159 |               10 |
| Labeled as 1 |                2 |              284 |

## Learning curves
![Learning curves](learning_curves.png)

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



[<< Go back](../README.md)
