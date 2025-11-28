# Summary of 5_Default_NeuralNetwork

[<< Go back](../README.md)


## Neural Network
- **n_jobs**: -1
- **dense_1_size**: 32
- **dense_2_size**: 16
- **learning_rate**: 0.05
- **explain_level**: 2

## Validation
 - **validation_type**: split
 - **train_ratio**: 0.75
 - **shuffle**: True
 - **stratify**: True

## Optimized metric
logloss

## Training time

1.1 seconds

## Metric details
|           |    score |     threshold |
|:----------|---------:|--------------:|
| logloss   | 0.141606 | nan           |
| auc       | 0.991402 | nan           |
| f1        | 0.965986 |   0.0485878   |
| accuracy  | 0.95614  |   0.0485878   |
| precision | 1        |   0.802339    |
| recall    | 1        |   3.36687e-15 |
| mcc       | 0.905824 |   0.0485878   |


## Metric details with threshold from accuracy metric
|           |    score |   threshold |
|:----------|---------:|------------:|
| logloss   | 0.141606 | nan         |
| auc       | 0.991402 | nan         |
| f1        | 0.965986 |   0.0485878 |
| accuracy  | 0.95614  |   0.0485878 |
| precision | 0.946667 |   0.0485878 |
| recall    | 0.986111 |   0.0485878 |
| mcc       | 0.905824 |   0.0485878 |


## Confusion matrix (at threshold=0.048588)
|              |   Predicted as 0 |   Predicted as 1 |
|:-------------|-----------------:|-----------------:|
| Labeled as 0 |               38 |                4 |
| Labeled as 1 |                1 |               71 |

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
