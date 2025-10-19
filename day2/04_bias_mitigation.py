"""
Day 2 - Module 4: Bias Mitigation
Topic: Understanding and Mitigating Algorithmic Bias

This module covers:
- Understanding algorithmic bias and fairness
- Fairness metrics and definitions
- Bias detection techniques
- Bias mitigation strategies (pre-processing, in-processing, post-processing)
- Fairness-aware machine learning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference
)
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
import warnings
warnings.filterwarnings('ignore')


class BiasDetector:
    """
    Toolkit for detecting bias in datasets and models.
    """
    
    def __init__(self, sensitive_features):
        """
        Initialize bias detector.
        
        Parameters:
        -----------
        sensitive_features : list
            List of sensitive feature names (e.g., ['gender', 'race'])
        """
        self.sensitive_features = sensitive_features
        self.metrics_results = {}
    
    def compute_group_statistics(self, data, target_col, sensitive_col):
        """
        Compute statistics by protected groups.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Dataset
        target_col : str
            Target column name
        sensitive_col : str
            Sensitive attribute column name
        
        Returns:
        --------
        pd.DataFrame : Group statistics
        """
        stats = data.groupby(sensitive_col)[target_col].agg([
            'count', 'mean', 'std'
        ]).reset_index()
        stats.columns = [sensitive_col, 'count', 'positive_rate', 'std']
        
        return stats
    
    def compute_fairness_metrics(self, y_true, y_pred, sensitive_features):
        """
        Compute various fairness metrics.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred : array-like
            Predicted labels
        sensitive_features : pd.DataFrame or dict
            Sensitive features
        
        Returns:
        --------
        dict : Fairness metrics
        """
        metrics = {}
        
        # Demographic Parity Difference
        dpd = demographic_parity_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        metrics['demographic_parity_difference'] = dpd
        
        # Demographic Parity Ratio
        dpr = demographic_parity_ratio(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        metrics['demographic_parity_ratio'] = dpr
        
        # Equalized Odds Difference
        eod = equalized_odds_difference(
            y_true, y_pred, sensitive_features=sensitive_features
        )
        metrics['equalized_odds_difference'] = eod
        
        # Detailed metrics by group
        metric_frame = MetricFrame(
            metrics=accuracy_score,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        metrics['accuracy_by_group'] = metric_frame.by_group.to_dict()
        
        return metrics
    
    def detect_disparate_impact(self, y_pred, sensitive_features, threshold=0.8):
        """
        Detect disparate impact using the 80% rule.
        
        Parameters:
        -----------
        y_pred : array-like
            Predicted labels
        sensitive_features : array-like
            Sensitive attribute values
        threshold : float, default=0.8
            Threshold for disparate impact (80% rule)
        
        Returns:
        --------
        dict : Disparate impact analysis
        """
        # Calculate selection rates by group
        df = pd.DataFrame({
            'prediction': y_pred,
            'group': sensitive_features
        })
        
        selection_rates = df.groupby('group')['prediction'].mean()
        
        # Calculate disparate impact ratio
        min_rate = selection_rates.min()
        max_rate = selection_rates.max()
        disparate_impact_ratio = min_rate / max_rate if max_rate > 0 else 0
        
        results = {
            'selection_rates': selection_rates.to_dict(),
            'disparate_impact_ratio': disparate_impact_ratio,
            'passes_80_rule': disparate_impact_ratio >= threshold
        }
        
        return results


class BiasMitigator:
    """
    Techniques for mitigating bias in machine learning models.
    """
    
    @staticmethod
    def reweigh_data(X, y, sensitive_features):
        """
        Pre-processing: Reweigh training data to mitigate bias.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Labels
        sensitive_features : pd.DataFrame
            Sensitive attributes
        
        Returns:
        --------
        np.array : Sample weights
        """
        # Create dataset for AIF360
        df = X.copy()
        df['label'] = y
        for col in sensitive_features.columns:
            df[col] = sensitive_features[col]
        
        # Convert to AIF360 format
        privileged_groups = [{sensitive_features.columns[0]: 1}]
        unprivileged_groups = [{sensitive_features.columns[0]: 0}]
        
        dataset = BinaryLabelDataset(
            df=df,
            label_names=['label'],
            protected_attribute_names=list(sensitive_features.columns)
        )
        
        # Apply reweighing
        RW = Reweighing(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups
        )
        dataset_transformed = RW.fit_transform(dataset)
        
        return dataset_transformed.instance_weights
    
    @staticmethod
    def train_fair_classifier(X_train, y_train, sensitive_features, 
                             constraint_type='demographic_parity'):
        """
        In-processing: Train a fairness-constrained classifier.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        sensitive_features : array-like
            Sensitive attributes
        constraint_type : str, default='demographic_parity'
            Type of fairness constraint: 'demographic_parity' or 'equalized_odds'
        
        Returns:
        --------
        model : Trained fair classifier
        """
        # Choose constraint
        if constraint_type == 'demographic_parity':
            constraint = DemographicParity()
        else:
            constraint = EqualizedOdds()
        
        # Base classifier
        base_classifier = LogisticRegression(max_iter=1000, random_state=42)
        
        # Fair classifier using Exponentiated Gradient
        fair_classifier = ExponentiatedGradient(
            base_classifier,
            constraints=constraint
        )
        
        fair_classifier.fit(X_train, y_train, sensitive_features=sensitive_features)
        
        return fair_classifier
    
    @staticmethod
    def threshold_optimizer(y_true, y_pred_proba, sensitive_features):
        """
        Post-processing: Optimize classification thresholds by group.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        sensitive_features : array-like
            Sensitive attributes
        
        Returns:
        --------
        dict : Optimal thresholds by group
        """
        df = pd.DataFrame({
            'y_true': y_true,
            'y_pred_proba': y_pred_proba,
            'group': sensitive_features
        })
        
        thresholds = {}
        
        for group in df['group'].unique():
            group_data = df[df['group'] == group]
            
            # Find threshold that maximizes accuracy for this group
            best_threshold = 0.5
            best_accuracy = 0
            
            for threshold in np.arange(0.1, 0.9, 0.05):
                y_pred = (group_data['y_pred_proba'] >= threshold).astype(int)
                acc = accuracy_score(group_data['y_true'], y_pred)
                
                if acc > best_accuracy:
                    best_accuracy = acc
                    best_threshold = threshold
            
            thresholds[group] = best_threshold
        
        return thresholds


# Demonstrations

def demonstrate_bias_detection():
    """
    Demonstrate bias detection in a dataset and model.
    """
    print("=" * 80)
    print("BIAS DETECTION DEMONSTRATION")
    print("=" * 80)
    
    # Create synthetic dataset with bias
    np.random.seed(42)
    n_samples = 1000
    
    # Protected attribute (e.g., gender: 0=female, 1=male)
    protected_attr = np.random.binomial(1, 0.5, n_samples)
    
    # Features with bias
    X = np.random.randn(n_samples, 5)
    
    # Biased target: more likely to be positive for protected group
    y = ((X[:, 0] + X[:, 1] + 0.5 * protected_attr + np.random.randn(n_samples) * 0.3) > 0).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['protected_attr'] = protected_attr
    df['target'] = y
    
    print(f"\n1. Dataset: {len(df)} samples")
    
    # Compute group statistics
    detector = BiasDetector(['protected_attr'])
    stats = detector.compute_group_statistics(df, 'target', 'protected_attr')
    
    print(f"\n2. Target Distribution by Protected Attribute:")
    print(stats.to_string(index=False))
    
    # Train a biased model
    X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
        df[[f'feature_{i}' for i in range(5)]],
        df['target'],
        df['protected_attr'],
        test_size=0.3,
        random_state=42
    )
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute fairness metrics
    fairness_metrics = detector.compute_fairness_metrics(
        y_test, y_pred, protected_test
    )
    
    print(f"\n3. Fairness Metrics:")
    print(f"   Demographic Parity Difference: {fairness_metrics['demographic_parity_difference']:.4f}")
    print(f"   Demographic Parity Ratio: {fairness_metrics['demographic_parity_ratio']:.4f}")
    print(f"   Equalized Odds Difference: {fairness_metrics['equalized_odds_difference']:.4f}")
    
    print(f"\n4. Accuracy by Group:")
    for group, acc in fairness_metrics['accuracy_by_group'].items():
        print(f"   Group {group}: {acc:.4f}")
    
    # Disparate impact
    impact = detector.detect_disparate_impact(y_pred, protected_test)
    print(f"\n5. Disparate Impact Analysis:")
    print(f"   Selection Rates: {impact['selection_rates']}")
    print(f"   Disparate Impact Ratio: {impact['disparate_impact_ratio']:.4f}")
    print(f"   Passes 80% Rule: {impact['passes_80_rule']}")
    
    print("\n" + "=" * 80)


def demonstrate_bias_mitigation():
    """
    Demonstrate bias mitigation techniques.
    """
    print("\n" + "=" * 80)
    print("BIAS MITIGATION DEMONSTRATION")
    print("=" * 80)
    
    # Create synthetic biased dataset
    np.random.seed(42)
    n_samples = 1000
    
    protected_attr = np.random.binomial(1, 0.5, n_samples)
    X = np.random.randn(n_samples, 5)
    y = ((X[:, 0] + X[:, 1] + 0.5 * protected_attr + np.random.randn(n_samples) * 0.3) > 0).astype(int)
    
    X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
        X, y, protected_attr, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {len(X)} samples")
    
    # Baseline model (biased)
    print(f"\n2. Baseline Model (No Mitigation):")
    baseline_model = LogisticRegression(random_state=42)
    baseline_model.fit(X_train, y_train)
    baseline_pred = baseline_model.predict(X_test)
    
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_dpd = demographic_parity_difference(
        y_test, baseline_pred, sensitive_features=protected_test
    )
    
    print(f"   Accuracy: {baseline_acc:.4f}")
    print(f"   Demographic Parity Difference: {baseline_dpd:.4f}")
    
    # Fair classifier
    print(f"\n3. Fair Classifier (Demographic Parity Constraint):")
    fair_model = BiasMitigator.train_fair_classifier(
        X_train, y_train, protected_train,
        constraint_type='demographic_parity'
    )
    fair_pred = fair_model.predict(X_test)
    
    fair_acc = accuracy_score(y_test, fair_pred)
    fair_dpd = demographic_parity_difference(
        y_test, fair_pred, sensitive_features=protected_test
    )
    
    print(f"   Accuracy: {fair_acc:.4f}")
    print(f"   Demographic Parity Difference: {fair_dpd:.4f}")
    
    print(f"\n4. Comparison:")
    print(f"   Accuracy drop: {baseline_acc - fair_acc:.4f}")
    print(f"   Fairness improvement: {abs(baseline_dpd) - abs(fair_dpd):.4f}")
    
    print("\n5. Trade-offs:")
    print("   - Bias mitigation often reduces overall accuracy")
    print("   - Improves fairness across protected groups")
    print("   - Choice depends on application requirements")
    print("   - Balance between performance and fairness")
    
    print("\n" + "=" * 80)


def demonstrate_fairness_definitions():
    """
    Explain different fairness definitions.
    """
    print("\n" + "=" * 80)
    print("FAIRNESS DEFINITIONS AND CONCEPTS")
    print("=" * 80)
    
    definitions = {
        "Demographic Parity": {
            "definition": "Equal selection rates across groups",
            "formula": "P(Ŷ=1|A=0) = P(Ŷ=1|A=1)",
            "use_case": "Hiring, loan approval where equal opportunity is priority"
        },
        "Equalized Odds": {
            "definition": "Equal true positive and false positive rates",
            "formula": "P(Ŷ=1|Y=y,A=0) = P(Ŷ=1|Y=y,A=1) for y∈{0,1}",
            "use_case": "Criminal justice, medical diagnosis"
        },
        "Equal Opportunity": {
            "definition": "Equal true positive rates across groups",
            "formula": "P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)",
            "use_case": "When false negatives are more costly than false positives"
        },
        "Predictive Parity": {
            "definition": "Equal positive predictive value across groups",
            "formula": "P(Y=1|Ŷ=1,A=0) = P(Y=1|Ŷ=1,A=1)",
            "use_case": "When confidence in positive predictions matters"
        }
    }
    
    print("\n1. Common Fairness Definitions:\n")
    for name, details in definitions.items():
        print(f"   {name}:")
        print(f"      Definition: {details['definition']}")
        print(f"      Formula: {details['formula']}")
        print(f"      Use Case: {details['use_case']}")
        print()
    
    print("2. Important Notes:")
    print("   - No single fairness definition fits all scenarios")
    print("   - Some fairness criteria are mutually exclusive")
    print("   - Context and stakeholder input are crucial")
    print("   - Legal and ethical considerations vary by domain")
    
    print("\n3. Mitigation Strategies:")
    print("   Pre-processing: Transform data to remove bias")
    print("   In-processing: Add fairness constraints to training")
    print("   Post-processing: Adjust predictions to achieve fairness")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_bias_detection()
    demonstrate_bias_mitigation()
    demonstrate_fairness_definitions()
    
    print("\n✅ Module 4 Complete: Bias Mitigation")
    print("\nKey Takeaways:")
    print("1. Algorithmic bias can perpetuate or amplify existing societal biases")
    print("2. Multiple fairness definitions exist, choose based on context")
    print("3. Bias can be mitigated at different stages: pre, in, or post-processing")
    print("4. There's often a trade-off between fairness and accuracy")
    print("5. Fairness considerations should be integrated throughout ML lifecycle")
    print("6. Regular auditing and monitoring are essential")
