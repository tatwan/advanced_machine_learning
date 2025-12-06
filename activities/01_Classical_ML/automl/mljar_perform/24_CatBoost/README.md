# Summary of 24_CatBoost

[<< Go back](../README.md)


## CatBoost
- **n_jobs**: -1
- **learning_rate**: 0.2
- **depth**: 5
- **rsm**: 1
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

3.0 seconds

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0915629 | nan           |
| auc       | 0.993297  | nan           |
| f1        | 0.979021  |   0.600721    |
| accuracy  | 0.973626  |   0.600721    |
| precision | 1         |   0.998032    |
| recall    | 1         |   3.54775e-05 |
| mcc       | 0.943518  |   0.600721    |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0915629 |  nan        |
| auc       | 0.993297  |  nan        |
| f1        | 0.979021  |    0.600721 |
| accuracy  | 0.973626  |    0.600721 |
| precision | 0.979021  |    0.600721 |
| recall    | 0.979021  |    0.600721 |
| mcc       | 0.943518  |    0.600721 |


## Confusion matrix (at threshold=0.600721)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |              163 |                6 |
| Labeled as 1 |                6 |              280 |

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
