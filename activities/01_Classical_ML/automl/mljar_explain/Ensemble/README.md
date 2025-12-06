# Summary of Ensemble

[<< Go back](../README.md)


## Ensemble structure
| Model                   |   Weight |
|:------------------------|---------:|
| 4_Default_Xgboost       |        2 |
| 5_Default_NeuralNetwork |        1 |

## Metric details
|           |     score |     threshold |
|:----------|----------:|--------------:|
| logloss   | 0.0820262 | nan           |
| auc       | 0.995701  | nan           |
| f1        | 0.97931   |   0.510947    |
| accuracy  | 0.973684  |   0.510947    |
| precision | 1         |   0.752732    |
| recall    | 1         |   0.000809835 |
| mcc       | 0.94334   |   0.510947    |


## Metric details with threshold from accuracy metric
|           |     score |   threshold |
|:----------|----------:|------------:|
| logloss   | 0.0820262 |  nan        |
| auc       | 0.995701  |  nan        |
| f1        | 0.97931   |    0.510947 |
| accuracy  | 0.973684  |    0.510947 |
| precision | 0.972603  |    0.510947 |
| recall    | 0.986111  |    0.510947 |
| mcc       | 0.94334   |    0.510947 |


## Confusion matrix (at threshold=0.510947)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |               40 |                2 |
| Labeled as 1 |                1 |               71 |

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
