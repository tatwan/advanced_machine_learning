# Summary of Ensemble

[<< Go back](../README.md)


## Ensemble structure
| Model                        |   Weight |
|:-----------------------------|---------:|
| 21_LightGBM_SelectedFeatures |       27 |
| 32_CatBoost                  |       23 |

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0422526 | nan           |
| auc       | 1         | nan           |
| f1        | 1         |   0.651791    |
| accuracy  | 1         |   0.651791    |
| precision | 1         |   0.651791    |
| recall    | 1         |   2.89103e-05 |
| mcc       | 1         |   0.651791    |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0422526 |  nan        |
| auc       | 1         |  nan        |
| f1        | 1         |    0.651791 |
| accuracy  | 1         |    0.651791 |
| precision | 1         |    0.651791 |
| recall    | 1         |    0.651791 |
| mcc       | 1         |    0.651791 |


## Confusion matrix (at threshold=0.651791)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |               17 |                0 |
| Labeled as 1 |                0 |               29 |

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
