"""
Day 2 - Module 1: Evaluation Metrics
Topic: Comprehensive Model Evaluation

This module covers:
- Classification metrics (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Regression metrics (RMSE, MAE, R², MAPE)
- Multi-class and multi-label metrics
- Custom metrics and business-specific evaluation
- Metric selection guidelines
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error,
    log_loss, matthews_corrcoef, cohen_kappa_score
)
from sklearn.model_selection import cross_val_predict
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class ClassificationMetrics:
    """
    Comprehensive classification metrics evaluation toolkit.
    """
    
    def __init__(self, y_true, y_pred, y_pred_proba=None):
        """
        Initialize with true labels and predictions.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        y_pred_proba : array-like, optional
            Predicted probabilities for positive class
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_pred_proba = y_pred_proba
    
    def binary_classification_metrics(self):
        """
        Compute comprehensive binary classification metrics.
        
        Returns:
        --------
        dict : Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
        metrics['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(self.y_true, self.y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(self.y_true, self.y_pred).ravel()
        metrics['true_negatives'] = tn
        metrics['false_positives'] = fp
        metrics['false_negatives'] = fn
        metrics['true_positives'] = tp
        
        # Specificity
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Additional metrics
        metrics['matthews_corrcoef'] = matthews_corrcoef(self.y_true, self.y_pred)
        metrics['cohen_kappa'] = cohen_kappa_score(self.y_true, self.y_pred)
        
        # Probability-based metrics
        if self.y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba)
            metrics['log_loss'] = log_loss(self.y_true, self.y_pred_proba)
            
            # PR-AUC
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_pred_proba)
            metrics['pr_auc'] = auc(recall, precision)
        
        return metrics
    
    def multiclass_classification_metrics(self, average='weighted'):
        """
        Compute multi-class classification metrics.
        
        Parameters:
        -----------
        average : str, default='weighted'
            Averaging method: 'micro', 'macro', 'weighted'
        
        Returns:
        --------
        dict : Dictionary of metrics
        """
        metrics = {}
        
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics[f'precision_{average}'] = precision_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )
        metrics[f'recall_{average}'] = recall_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )
        metrics[f'f1_score_{average}'] = f1_score(
            self.y_true, self.y_pred, average=average, zero_division=0
        )
        
        return metrics
    
    def get_confusion_matrix(self):
        """
        Get confusion matrix.
        
        Returns:
        --------
        np.array : Confusion matrix
        """
        return confusion_matrix(self.y_true, self.y_pred)
    
    def get_classification_report(self):
        """
        Get detailed classification report.
        
        Returns:
        --------
        str : Classification report
        """
        return classification_report(self.y_true, self.y_pred)


class RegressionMetrics:
    """
    Comprehensive regression metrics evaluation toolkit.
    """
    
    def __init__(self, y_true, y_pred):
        """
        Initialize with true values and predictions.
        
        Parameters:
        -----------
        y_true : array-like
            True values
        y_pred : array-like
            Predicted values
        """
        self.y_true = y_true
        self.y_pred = y_pred
    
    def compute_all_metrics(self):
        """
        Compute comprehensive regression metrics.
        
        Returns:
        --------
        dict : Dictionary of metrics
        """
        metrics = {}
        
        # Mean Squared Error and RMSE
        mse = mean_squared_error(self.y_true, self.y_pred)
        metrics['mse'] = mse
        metrics['rmse'] = np.sqrt(mse)
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(self.y_true, self.y_pred)
        
        # R-squared
        metrics['r2_score'] = r2_score(self.y_true, self.y_pred)
        
        # Mean Absolute Percentage Error
        # Handle division by zero
        mask = self.y_true != 0
        if mask.sum() > 0:
            mape = np.mean(np.abs((self.y_true[mask] - self.y_pred[mask]) / self.y_true[mask])) * 100
            metrics['mape'] = mape
        else:
            metrics['mape'] = np.nan
        
        # Adjusted R-squared (requires number of features)
        # For demonstration, we'll skip this or it can be added with n_features parameter
        
        # Max Error
        metrics['max_error'] = np.max(np.abs(self.y_true - self.y_pred))
        
        # Median Absolute Error
        metrics['median_absolute_error'] = np.median(np.abs(self.y_true - self.y_pred))
        
        return metrics
    
    def compute_residuals(self):
        """
        Compute residuals.
        
        Returns:
        --------
        np.array : Residuals (y_true - y_pred)
        """
        return self.y_true - self.y_pred
    
    def residual_statistics(self):
        """
        Compute residual statistics.
        
        Returns:
        --------
        dict : Residual statistics
        """
        residuals = self.compute_residuals()
        
        stats = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'median_residual': np.median(residuals)
        }
        
        return stats


class CustomMetrics:
    """
    Custom and business-specific metrics.
    """
    
    @staticmethod
    def cost_sensitive_accuracy(y_true, y_pred, fp_cost=1, fn_cost=1):
        """
        Accuracy weighted by misclassification costs.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        fp_cost : float, default=1
            Cost of false positive
        fn_cost : float, default=1
            Cost of false negative
        
        Returns:
        --------
        float : Cost-weighted score
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_cost = fp * fp_cost + fn * fn_cost
        max_cost = len(y_true) * max(fp_cost, fn_cost)
        
        # Normalize to 0-1 range (higher is better)
        score = 1 - (total_cost / max_cost)
        
        return score
    
    @staticmethod
    def profit_score(y_true, y_pred, tp_profit=10, tn_profit=0, fp_cost=-5, fn_cost=-8):
        """
        Calculate profit-based score for business scenarios.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        tp_profit : float, default=10
            Profit from true positive
        tn_profit : float, default=0
            Profit from true negative
        fp_cost : float, default=-5
            Cost of false positive
        fn_cost : float, default=-8
            Cost of false negative
        
        Returns:
        --------
        float : Total profit
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        total_profit = (tp * tp_profit + 
                       tn * tn_profit + 
                       fp * fp_cost + 
                       fn * fn_cost)
        
        return total_profit
    
    @staticmethod
    def top_k_accuracy(y_true, y_pred_proba, k=3):
        """
        Top-K accuracy for multi-class problems.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities (n_samples, n_classes)
        k : int, default=3
            Number of top predictions to consider
        
        Returns:
        --------
        float : Top-K accuracy
        """
        top_k_preds = np.argsort(y_pred_proba, axis=1)[:, -k:]
        correct = sum([y_true[i] in top_k_preds[i] for i in range(len(y_true))])
        
        return correct / len(y_true)


# Demonstrations

def demonstrate_classification_metrics():
    """
    Demonstrate classification metrics evaluation.
    """
    print("=" * 80)
    print("CLASSIFICATION METRICS DEMONSTRATION")
    print("=" * 80)
    
    # Generate imbalanced dataset
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, weights=[0.7, 0.3], random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset Information:")
    print(f"   Total samples: {len(y)}")
    print(f"   Class distribution: {np.bincount(y)}")
    print(f"   Imbalance ratio: {np.bincount(y)[0] / np.bincount(y)[1]:.2f}")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    evaluator = ClassificationMetrics(y_test, y_pred, y_pred_proba)
    metrics = evaluator.binary_classification_metrics()
    
    print(f"\n2. Binary Classification Metrics:")
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")
    print(f"   F1-Score: {metrics['f1_score']:.4f}")
    print(f"   Specificity: {metrics['specificity']:.4f}")
    print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"   PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"   Matthews Correlation: {metrics['matthews_corrcoef']:.4f}")
    
    print(f"\n3. Confusion Matrix:")
    print(f"   True Negatives: {metrics['true_negatives']}")
    print(f"   False Positives: {metrics['false_positives']}")
    print(f"   False Negatives: {metrics['false_negatives']}")
    print(f"   True Positives: {metrics['true_positives']}")
    
    print("\n4. Classification Report:")
    print(evaluator.get_classification_report())
    
    print("=" * 80)


def demonstrate_regression_metrics():
    """
    Demonstrate regression metrics evaluation.
    """
    print("\n" + "=" * 80)
    print("REGRESSION METRICS DEMONSTRATION")
    print("=" * 80)
    
    # Generate regression dataset
    X, y = make_regression(
        n_samples=1000, n_features=20, n_informative=15,
        noise=10, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset Information:")
    print(f"   Total samples: {len(y)}")
    print(f"   Features: {X.shape[1]}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    evaluator = RegressionMetrics(y_test, y_pred)
    metrics = evaluator.compute_all_metrics()
    
    print(f"\n2. Regression Metrics:")
    print(f"   MSE: {metrics['mse']:.4f}")
    print(f"   RMSE: {metrics['rmse']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   R² Score: {metrics['r2_score']:.4f}")
    print(f"   MAPE: {metrics['mape']:.4f}%")
    print(f"   Max Error: {metrics['max_error']:.4f}")
    print(f"   Median Absolute Error: {metrics['median_absolute_error']:.4f}")
    
    # Residual analysis
    residual_stats = evaluator.residual_statistics()
    
    print(f"\n3. Residual Analysis:")
    print(f"   Mean Residual: {residual_stats['mean_residual']:.4f}")
    print(f"   Std Residual: {residual_stats['std_residual']:.4f}")
    print(f"   Min Residual: {residual_stats['min_residual']:.4f}")
    print(f"   Max Residual: {residual_stats['max_residual']:.4f}")
    
    print("\n" + "=" * 80)


def demonstrate_custom_metrics():
    """
    Demonstrate custom and business-specific metrics.
    """
    print("\n" + "=" * 80)
    print("CUSTOM METRICS DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=20,
        weights=[0.6, 0.4], random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print("\n1. Business Scenario: Fraud Detection")
    print("   - True Positive (catch fraud): +$100")
    print("   - True Negative (correct acceptance): $0")
    print("   - False Positive (false alarm): -$20")
    print("   - False Negative (miss fraud): -$500")
    
    # Calculate profit
    profit = CustomMetrics.profit_score(
        y_test, y_pred,
        tp_profit=100, tn_profit=0,
        fp_cost=-20, fn_cost=-500
    )
    
    print(f"\n2. Profit-based Evaluation:")
    print(f"   Total Profit: ${profit:.2f}")
    print(f"   Profit per sample: ${profit/len(y_test):.2f}")
    
    # Cost-sensitive accuracy
    cost_acc = CustomMetrics.cost_sensitive_accuracy(
        y_test, y_pred,
        fp_cost=1, fn_cost=25  # FN is 25x more costly
    )
    
    print(f"\n3. Cost-Sensitive Metrics:")
    print(f"   Standard Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   Cost-Sensitive Score: {cost_acc:.4f}")
    
    print("\n" + "=" * 80)


def demonstrate_metric_selection():
    """
    Provide guidelines for metric selection based on problem characteristics.
    """
    print("\n" + "=" * 80)
    print("METRIC SELECTION GUIDELINES")
    print("=" * 80)
    
    guidelines = {
        "Balanced Classification": [
            "Accuracy",
            "F1-Score",
            "ROC-AUC"
        ],
        "Imbalanced Classification": [
            "Precision-Recall AUC",
            "F1-Score",
            "Matthews Correlation Coefficient"
        ],
        "Cost-Sensitive Problems": [
            "Custom Profit/Cost metrics",
            "Weighted F1-Score",
            "Cost-sensitive accuracy"
        ],
        "Ranking Problems": [
            "NDCG (Normalized Discounted Cumulative Gain)",
            "MAP (Mean Average Precision)",
            "Top-K Accuracy"
        ],
        "Regression (General)": [
            "RMSE (for outlier sensitivity)",
            "MAE (for robustness)",
            "R² (for variance explained)"
        ],
        "Regression (Business)": [
            "MAPE (for percentage errors)",
            "Custom business metrics",
            "Quantile loss (for specific predictions)"
        ]
    }
    
    print("\n1. Recommended Metrics by Problem Type:\n")
    for problem_type, metrics_list in guidelines.items():
        print(f"   {problem_type}:")
        for metric in metrics_list:
            print(f"      - {metric}")
        print()
    
    print("2. Key Considerations:")
    print("   - Class imbalance → Avoid accuracy, use F1 or PR-AUC")
    print("   - Cost-sensitive → Use custom metrics reflecting business value")
    print("   - Multi-class → Use macro/weighted averaging")
    print("   - Outliers matter → Use MSE/RMSE")
    print("   - Outliers don't matter → Use MAE")
    print("   - Need interpretability → Use simple metrics (accuracy, MAE)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_classification_metrics()
    demonstrate_regression_metrics()
    demonstrate_custom_metrics()
    demonstrate_metric_selection()
    
    print("\n✅ Module 1 Complete: Evaluation Metrics")
    print("\nKey Takeaways:")
    print("1. Choose metrics appropriate for your problem and data distribution")
    print("2. Use multiple metrics to get a complete picture of performance")
    print("3. Consider business impact when selecting metrics")
    print("4. PR-AUC is better than ROC-AUC for imbalanced datasets")
    print("5. Always analyze confusion matrix in addition to summary metrics")
