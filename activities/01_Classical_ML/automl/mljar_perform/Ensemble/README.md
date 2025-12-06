# Summary of Ensemble

[<< Go back](../README.md)


## Ensemble structure
| Model       |   Weight |
|:------------|---------:|
| 14_CatBoost |        2 |
| 23_CatBoost |        8 |
| 9_LightGBM  |        9 |

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0823791 | nan           |
| auc       | 0.994517  | nan           |
| f1        | 0.976027  |   0.325462    |
| accuracy  | 0.969231  |   0.325462    |
| precision | 1         |   0.99258     |
| recall    | 1         |   6.48632e-05 |
| mcc       | 0.935387  |   0.81011     |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0823791 |  nan        |
| auc       | 0.994517  |  nan        |
| f1        | 0.976027  |    0.325462 |
| accuracy  | 0.969231  |    0.325462 |
| precision | 0.956376  |    0.325462 |
| recall    | 0.996503  |    0.325462 |
| mcc       | 0.93467   |    0.325462 |


## Confusion matrix (at threshold=0.325462)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |              156 |               13 |
| Labeled as 1 |                1 |              285 |

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
