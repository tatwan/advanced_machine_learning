"""
Day 3 - Module 3: LIME Explainability
Topic: Local Interpretable Model-agnostic Explanations (LIME)

This module covers:
- LIME theory and methodology
- LIME for tabular data
- LIME for text and images (concepts)
- Comparison with SHAP
- Practical applications
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')


class LIMEAnalyzer:
    """
    LIME-based model interpretability toolkit.
    """
    
    def __init__(self, X_train, feature_names, class_names=None, categorical_features=None):
        """
        Initialize LIME analyzer.
        
        Parameters:
        -----------
        X_train : array-like
            Training data (used to understand feature distributions)
        feature_names : list
            Names of features
        class_names : list, optional
            Names of classes
        categorical_features : list, optional
            Indices of categorical features
        """
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=class_names,
            categorical_features=categorical_features,
            mode='classification'
        )
        self.feature_names = feature_names
    
    def explain_instance(self, model, instance, num_features=10):
        """
        Explain a single prediction.
        
        Parameters:
        -----------
        model : estimator
            Trained model with predict_proba method
        instance : array-like
            Single instance to explain
        num_features : int, default=10
            Number of features to include in explanation
        
        Returns:
        --------
        dict : Explanation dictionary
        """
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=num_features
        )
        
        # Extract feature importances
        feature_weights = explanation.as_list()
        
        # Get prediction
        prediction = model.predict_proba(instance.reshape(1, -1))[0]
        
        return {
            'feature_weights': feature_weights,
            'prediction': prediction,
            'explanation_obj': explanation
        }
    
    def get_feature_importance_df(self, explanation):
        """
        Convert LIME explanation to DataFrame.
        
        Parameters:
        -----------
        explanation : dict
            Explanation from explain_instance
        
        Returns:
        --------
        pd.DataFrame : Feature importance dataframe
        """
        feature_weights = explanation['feature_weights']
        
        df = pd.DataFrame(feature_weights, columns=['feature_rule', 'weight'])
        df['abs_weight'] = df['weight'].abs()
        df = df.sort_values('abs_weight', ascending=False)
        
        return df
    
    def batch_explain(self, model, X_test, num_samples=10, num_features=5):
        """
        Explain multiple instances.
        
        Parameters:
        -----------
        model : estimator
            Trained model
        X_test : array-like
            Test data
        num_samples : int, default=10
            Number of samples to explain
        num_features : int, default=5
            Number of features per explanation
        
        Returns:
        --------
        list : List of explanations
        """
        explanations = []
        
        for i in range(min(num_samples, len(X_test))):
            exp = self.explain_instance(model, X_test[i], num_features)
            explanations.append(exp)
        
        return explanations


class LIMEInterpretation:
    """
    Helper for interpreting and comparing LIME explanations.
    """
    
    @staticmethod
    def aggregate_feature_importance(explanations):
        """
        Aggregate feature importance across multiple explanations.
        
        Parameters:
        -----------
        explanations : list
            List of LIME explanations
        
        Returns:
        --------
        pd.DataFrame : Aggregated importance
        """
        all_features = {}
        
        for exp in explanations:
            for feature_rule, weight in exp['feature_weights']:
                # Extract feature name (before comparison operator)
                feature = feature_rule.split(' ')[0] if ' ' in feature_rule else feature_rule
                
                if feature not in all_features:
                    all_features[feature] = []
                all_features[feature].append(abs(weight))
        
        # Aggregate
        aggregated = []
        for feature, weights in all_features.items():
            aggregated.append({
                'feature': feature,
                'mean_importance': np.mean(weights),
                'std_importance': np.std(weights),
                'count': len(weights)
            })
        
        df = pd.DataFrame(aggregated).sort_values('mean_importance', ascending=False)
        return df
    
    @staticmethod
    def compare_shap_lime_importance(shap_importance, lime_importance):
        """
        Compare SHAP and LIME feature importance.
        
        Parameters:
        -----------
        shap_importance : pd.DataFrame
            SHAP feature importance
        lime_importance : pd.DataFrame
            LIME feature importance
        
        Returns:
        --------
        pd.DataFrame : Comparison dataframe
        """
        # Merge on feature name
        comparison = pd.merge(
            shap_importance[['feature', 'importance']],
            lime_importance[['feature', 'mean_importance']],
            on='feature',
            how='outer'
        ).fillna(0)
        
        comparison.columns = ['feature', 'shap_importance', 'lime_importance']
        
        # Normalize for comparison
        comparison['shap_rank'] = comparison['shap_importance'].rank(ascending=False)
        comparison['lime_rank'] = comparison['lime_importance'].rank(ascending=False)
        comparison['rank_diff'] = abs(comparison['shap_rank'] - comparison['lime_rank'])
        
        return comparison.sort_values('shap_importance', ascending=False)


# Demonstrations

def demonstrate_lime_basics():
    """
    Demonstrate basic LIME functionality.
    """
    print("=" * 80)
    print("LIME (Local Interpretable Model-agnostic Explanations) DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=7,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    class_names = ['Class 0', 'Class 1']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"   Model Accuracy: {accuracy:.4f}")
    
    # Create LIME analyzer
    print(f"\n2. Creating LIME Explainer...")
    analyzer = LIMEAnalyzer(X_train, feature_names, class_names)
    
    # Explain a single instance
    print(f"\n3. Explaining Single Instance:")
    instance_idx = 0
    explanation = analyzer.explain_instance(model, X_test[instance_idx])
    
    print(f"   Predicted Probabilities:")
    for i, class_name in enumerate(class_names):
        print(f"      {class_name}: {explanation['prediction'][i]:.4f}")
    
    print(f"\n   Top Feature Contributions:")
    importance_df = analyzer.get_feature_importance_df(explanation)
    print(importance_df.head(5).to_string(index=False))
    
    print("\n" + "=" * 80)


def demonstrate_lime_local_fidelity():
    """
    Demonstrate LIME's local fidelity concept.
    """
    print("\n" + "=" * 80)
    print("LIME LOCAL FIDELITY DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. Concept:")
    print("   LIME creates a simple interpretable model (linear) that")
    print("   approximates the complex model's behavior locally around")
    print("   the instance being explained.")
    
    print("\n2. Process:")
    print("   a) Perturb the instance by sampling around it")
    print("   b) Get predictions from complex model for perturbed samples")
    print("   c) Weight samples by proximity to original instance")
    print("   d) Train simple model on weighted samples")
    print("   e) Use simple model's weights as explanation")
    
    print("\n3. Local vs Global:")
    print("   - LIME is LOCAL: explains individual predictions")
    print("   - Different instances may have different explanations")
    print("   - This is appropriate for complex non-linear models")
    print("   - Global importance is different from local explanation")
    
    # Generate data
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=5,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Create analyzer
    analyzer = LIMEAnalyzer(X_train, feature_names)
    
    # Compare explanations for different instances
    print("\n4. Comparing Explanations for Different Instances:\n")
    
    for i in range(3):
        exp = analyzer.explain_instance(model, X_test[i], num_features=5)
        pred_class = np.argmax(exp['prediction'])
        
        print(f"   Instance {i+1} (Predicted: Class {pred_class}):")
        importance_df = analyzer.get_feature_importance_df(exp)
        print(f"      Top 3 features:")
        for _, row in importance_df.head(3).iterrows():
            print(f"         {row['feature_rule']}: {row['weight']:.4f}")
        print()
    
    print("\n" + "=" * 80)


def demonstrate_lime_vs_shap():
    """
    Compare LIME and SHAP explanations.
    """
    print("\n" + "=" * 80)
    print("LIME VS SHAP COMPARISON")
    print("=" * 80)
    
    comparison_table = {
        "Aspect": [
            "Theoretical Foundation",
            "Model Agnostic",
            "Local vs Global",
            "Computation Speed",
            "Consistency",
            "Feature Interactions",
            "Ease of Use"
        ],
        "LIME": [
            "Perturbation-based approximation",
            "Yes - works with any model",
            "Local explanations only",
            "Fast (sample-based)",
            "May vary across runs",
            "Limited support",
            "Simple, intuitive"
        ],
        "SHAP": [
            "Game theory (Shapley values)",
            "Yes - multiple explainers",
            "Both local and global",
            "Slower (especially Kernel SHAP)",
            "Consistent and unique",
            "Full support",
            "More complex, powerful"
        ]
    }
    
    comparison_df = pd.DataFrame(comparison_table)
    
    print("\n1. LIME vs SHAP Comparison:\n")
    for _, row in comparison_df.iterrows():
        print(f"   {row['Aspect']}:")
        print(f"      LIME: {row['LIME']}")
        print(f"      SHAP: {row['SHAP']}")
        print()
    
    print("2. When to Use LIME:")
    print("   - Quick local explanations needed")
    print("   - Working with text or image data")
    print("   - Model type doesn't have efficient SHAP explainer")
    print("   - Computational resources are limited")
    
    print("\n3. When to Use SHAP:")
    print("   - Theoretical guarantees are important")
    print("   - Global feature importance is needed")
    print("   - Feature interactions matter")
    print("   - Consistency across explanations is crucial")
    print("   - Working with tree-based models (TreeSHAP is fast)")
    
    print("\n4. Best Practice:")
    print("   - Use both for critical applications")
    print("   - SHAP for overall understanding")
    print("   - LIME for quick spot-checks")
    print("   - Compare results for validation")
    
    print("\n" + "=" * 80)


def demonstrate_lime_use_cases():
    """
    Demonstrate practical LIME use cases.
    """
    print("\n" + "=" * 80)
    print("LIME PRACTICAL USE CASES")
    print("=" * 80)
    
    use_cases = {
        "Healthcare": {
            "scenario": "Explaining medical diagnosis predictions",
            "example": "Why did the model predict high risk of disease?",
            "value": "Doctor can validate reasoning, catch potential errors"
        },
        "Finance": {
            "scenario": "Credit scoring and loan decisions",
            "example": "Why was this loan application denied?",
            "value": "Regulatory compliance, customer transparency"
        },
        "Text Classification": {
            "scenario": "Spam detection, sentiment analysis",
            "example": "Which words made this email spam?",
            "value": "Debug model, improve filters"
        },
        "E-commerce": {
            "scenario": "Product recommendation explanations",
            "example": "Why was this product recommended?",
            "value": "Build user trust, improve engagement"
        },
        "Fraud Detection": {
            "scenario": "Explaining fraud predictions",
            "example": "What triggered the fraud alert?",
            "value": "Reduce false positives, improve investigation"
        }
    }
    
    print("\n1. Real-World LIME Applications:\n")
    for domain, details in use_cases.items():
        print(f"   {domain}:")
        print(f"      Scenario: {details['scenario']}")
        print(f"      Example: {details['example']}")
        print(f"      Value: {details['value']}")
        print()
    
    print("2. Implementation Tips:")
    print("   - Start with a small number of features (5-10)")
    print("   - Use domain expertise to validate explanations")
    print("   - Test explanation stability with multiple runs")
    print("   - Communicate limitations to stakeholders")
    print("   - Consider computational cost in production")
    
    print("\n3. Common Pitfalls:")
    print("   - Over-interpreting local explanations globally")
    print("   - Ignoring feature correlation effects")
    print("   - Not validating with domain experts")
    print("   - Assuming explanations are always correct")
    print("   - Neglecting explanation variance")
    
    print("\n" + "=" * 80)


def demonstrate_lime_aggregation():
    """
    Demonstrate aggregating LIME explanations across instances.
    """
    print("\n" + "=" * 80)
    print("AGGREGATING LIME EXPLANATIONS")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=300, n_features=8, n_informative=5,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Create analyzer
    analyzer = LIMEAnalyzer(X_train, feature_names)
    
    # Explain multiple instances
    print("\n1. Generating Explanations for 20 Instances...")
    explanations = analyzer.batch_explain(model, X_test, num_samples=20, num_features=8)
    
    # Aggregate
    aggregated = LIMEInterpretation.aggregate_feature_importance(explanations)
    
    print(f"\n2. Aggregated Feature Importance:")
    print(aggregated.head(8).to_string(index=False))
    
    print("\n3. Interpretation:")
    print("   - Mean importance shows average impact across instances")
    print("   - Std importance shows consistency/variability")
    print("   - Count shows how often feature appears in explanations")
    print("   - This provides a 'global' view from local explanations")
    
    print("\n4. Caution:")
    print("   - Aggregated LIME ≠ True global importance")
    print("   - Only approximates global feature importance")
    print("   - Use SHAP for proper global importance")
    print("   - Good for exploratory analysis")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_lime_basics()
    demonstrate_lime_local_fidelity()
    demonstrate_lime_vs_shap()
    demonstrate_lime_use_cases()
    demonstrate_lime_aggregation()
    
    print("\n✅ Module 3 Complete: LIME Explainability")
    print("\nKey Takeaways:")
    print("1. LIME provides local, model-agnostic explanations")
    print("2. Based on local linear approximation of complex models")
    print("3. Fast and intuitive, but less theoretically rigorous than SHAP")
    print("4. Great for text and image data")
    print("5. Useful for quick explanations and debugging")
    print("6. Best used complementary to SHAP")
