# Summary of 18_RandomForest

[<< Go back](../README.md)


## Random Forest
- **n_jobs**: -1
- **criterion**: entropy
- **max_features**: 1.0
- **min_samples_split**: 20
- **max_depth**: 3
- **eval_metric_name**: logloss
- **explain_level**: 1

## Validation
 - **validation_type**: kfold
 - **k_folds**: 5
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
logloss

## Training time

5.0 seconds

## Metric details
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.139588 |  nan        |
| auc       | 0.987555 |  nan        |
| f1        | 0.958904 |    0.367614 |
| accuracy  | 0.947253 |    0.367614 |
| precision | 1        |    0.99003  |
| recall    | 1        |    0        |
| mcc       | 0.886829 |    0.367614 |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.139588 |  nan        |
| auc       | 0.987555 |  nan        |
| f1        | 0.958904 |    0.367614 |
| accuracy  | 0.947253 |    0.367614 |
| precision | 0.939597 |    0.367614 |
| recall    | 0.979021 |    0.367614 |
| mcc       | 0.886829 |    0.367614 |


## Confusion matrix (at threshold=0.367614)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |              151 |               18 |
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
