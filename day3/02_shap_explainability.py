"""
Day 3 - Module 2: SHAP Explainability
Topic: SHapley Additive exPlanations (SHAP)

This module covers:
- SHAP theory and Shapley values
- Different SHAP explainers
- Global and local interpretability
- Feature importance via SHAP
- Practical applications
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import shap
import warnings
warnings.filterwarnings('ignore')


class SHAPAnalyzer:
    """
    SHAP-based model interpretability toolkit.
    """
    
    def __init__(self, model, X_train):
        """
        Initialize SHAP analyzer.
        
        Parameters:
        -----------
        model : estimator
            Trained model
        X_train : array-like
            Training data (used as background for some explainers)
        """
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, explainer_type='tree'):
        """
        Create appropriate SHAP explainer for the model.
        
        Parameters:
        -----------
        explainer_type : str, default='tree'
            Type of explainer: 'tree', 'kernel', 'linear'
        
        Returns:
        --------
        self : SHAPAnalyzer
            Analyzer with explainer created
        """
        if explainer_type == 'tree':
            # Fast for tree-based models
            self.explainer = shap.TreeExplainer(self.model)
        elif explainer_type == 'kernel':
            # Model-agnostic but slower
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                shap.sample(self.X_train, 100)
            )
        elif explainer_type == 'linear':
            # For linear models
            self.explainer = shap.LinearExplainer(
                self.model,
                self.X_train
            )
        else:
            raise ValueError(f"Unknown explainer type: {explainer_type}")
        
        return self
    
    def explain_predictions(self, X_test):
        """
        Compute SHAP values for test data.
        
        Parameters:
        -----------
        X_test : array-like
            Test data to explain
        
        Returns:
        --------
        np.array : SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer must be created first")
        
        self.shap_values = self.explainer.shap_values(X_test)
        
        return self.shap_values
    
    def get_global_importance(self, feature_names):
        """
        Get global feature importance based on mean absolute SHAP values.
        
        Parameters:
        -----------
        feature_names : list
            Names of features
        
        Returns:
        --------
        pd.DataFrame : Feature importance ranking
        """
        if self.shap_values is None:
            raise ValueError("SHAP values must be computed first")
        
        # Handle multi-class output
        shap_vals = self.shap_values
        if isinstance(shap_vals, list):
            # For multi-class, average across classes
            shap_vals = np.abs(shap_vals).mean(axis=0)
        
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def explain_single_prediction(self, X_sample, feature_names):
        """
        Explain a single prediction.
        
        Parameters:
        -----------
        X_sample : array-like
            Single sample to explain
        feature_names : list
            Names of features
        
        Returns:
        --------
        pd.DataFrame : Feature contributions
        """
        if len(X_sample.shape) == 1:
            X_sample = X_sample.reshape(1, -1)
        
        shap_vals = self.explainer.shap_values(X_sample)
        
        # Handle multi-class
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # Use positive class for binary
        
        explanation = pd.DataFrame({
            'feature': feature_names,
            'value': X_sample[0],
            'shap_value': shap_vals[0],
            'abs_shap': np.abs(shap_vals[0])
        }).sort_values('abs_shap', ascending=False)
        
        return explanation
    
    def get_feature_interactions(self, X_test):
        """
        Compute SHAP interaction values.
        
        Parameters:
        -----------
        X_test : array-like
            Test data
        
        Returns:
        --------
        np.array : Interaction values
        """
        if hasattr(self.explainer, 'shap_interaction_values'):
            return self.explainer.shap_interaction_values(X_test)
        else:
            print("Interaction values not available for this explainer type")
            return None


class SHAPVisualizations:
    """
    Helper for SHAP visualizations and interpretations.
    """
    
    @staticmethod
    def interpret_shap_values(shap_values, feature_names, top_n=5):
        """
        Provide text interpretation of SHAP values.
        
        Parameters:
        -----------
        shap_values : np.array
            SHAP values for a prediction
        feature_names : list
            Names of features
        top_n : int, default=5
            Number of top features to interpret
        
        Returns:
        --------
        str : Interpretation text
        """
        # Get top positive and negative contributors
        sorted_indices = np.argsort(np.abs(shap_values))[::-1][:top_n]
        
        interpretation = []
        interpretation.append("Top Feature Contributions:")
        
        for idx in sorted_indices:
            feature = feature_names[idx]
            shap_val = shap_values[idx]
            
            if shap_val > 0:
                direction = "increases"
            else:
                direction = "decreases"
            
            interpretation.append(
                f"  - {feature}: {direction} prediction by {abs(shap_val):.4f}"
            )
        
        return "\n".join(interpretation)


# Demonstrations

def demonstrate_shap_basics():
    """
    Demonstrate basic SHAP functionality.
    """
    print("=" * 80)
    print("SHAP (SHapley Additive exPlanations) DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=10, n_informative=7,
        n_redundant=2, random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"   Model Accuracy: {accuracy:.4f}")
    
    # Create SHAP analyzer
    print(f"\n2. Creating SHAP Explainer...")
    analyzer = SHAPAnalyzer(model, X_train)
    analyzer.create_explainer(explainer_type='tree')
    
    # Compute SHAP values
    print(f"   Computing SHAP values...")
    analyzer.explain_predictions(X_test)
    
    # Global importance
    print(f"\n3. Global Feature Importance (Mean |SHAP|):")
    importance = analyzer.get_global_importance(feature_names)
    print(importance.head(10).to_string(index=False))
    
    print("\n" + "=" * 80)


def demonstrate_local_explanations():
    """
    Demonstrate local (instance-level) explanations with SHAP.
    """
    print("\n" + "=" * 80)
    print("LOCAL SHAP EXPLANATIONS DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=500, n_features=8, n_informative=6,
        random_state=42
    )
    
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train model
    model = GradientBoostingClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"\n1. Explaining Individual Predictions")
    print(f"   Model: Gradient Boosting Classifier")
    
    # Create analyzer
    analyzer = SHAPAnalyzer(model, X_train)
    analyzer.create_explainer(explainer_type='tree')
    analyzer.explain_predictions(X_test)
    
    # Explain first few predictions
    for i in range(3):
        print(f"\n2. Sample {i+1}:")
        X_sample = X_test[i]
        prediction = model.predict_proba(X_sample.reshape(1, -1))[0]
        
        print(f"   Predicted Probability: Class 0: {prediction[0]:.4f}, Class 1: {prediction[1]:.4f}")
        
        explanation = analyzer.explain_single_prediction(X_sample, feature_names)
        print(f"\n   Top 5 Feature Contributions:")
        print(explanation.head(5)[['feature', 'value', 'shap_value']].to_string(index=False))
        
        # Interpret
        shap_vals = explanation['shap_value'].values
        interpretation = SHAPVisualizations.interpret_shap_values(
            shap_vals, feature_names, top_n=3
        )
        print(f"\n   {interpretation}")
    
    print("\n" + "=" * 80)


def demonstrate_shap_use_cases():
    """
    Demonstrate practical use cases for SHAP.
    """
    print("\n" + "=" * 80)
    print("SHAP PRACTICAL USE CASES")
    print("=" * 80)
    
    use_cases = {
        "Model Debugging": {
            "description": "Identify unexpected feature contributions",
            "example": "SHAP reveals model uses irrelevant date field for predictions",
            "action": "Remove or encode date properly"
        },
        "Regulatory Compliance": {
            "description": "Explain individual predictions for regulated industries",
            "example": "Loan application - explain why customer was denied",
            "action": "Provide SHAP-based explanations to customers"
        },
        "Feature Engineering": {
            "description": "Discover important feature interactions",
            "example": "SHAP interactions show age*income is predictive",
            "action": "Create explicit interaction features"
        },
        "Model Comparison": {
            "description": "Compare feature importance across models",
            "example": "Random Forest vs XGBoost - which features do they rely on?",
            "action": "Choose model with more aligned feature importance"
        },
        "Bias Detection": {
            "description": "Identify unwanted reliance on protected attributes",
            "example": "SHAP shows gender has high importance in hiring model",
            "action": "Retrain model with fairness constraints"
        }
    }
    
    print("\n1. Common SHAP Use Cases:\n")
    for use_case, details in use_cases.items():
        print(f"   {use_case}:")
        print(f"      Description: {details['description']}")
        print(f"      Example: {details['example']}")
        print(f"      Action: {details['action']}")
        print()
    
    print("2. Advantages of SHAP:")
    print("   - Theoretically grounded (Shapley values from game theory)")
    print("   - Consistent and locally accurate")
    print("   - Provides both global and local explanations")
    print("   - Works with any model type")
    print("   - Handles feature interactions")
    
    print("\n3. Limitations:")
    print("   - Can be computationally expensive")
    print("   - Kernel SHAP requires sampling (approximate)")
    print("   - Correlation between features can complicate interpretation")
    print("   - Requires careful communication to non-technical stakeholders")
    
    print("\n" + "=" * 80)


def demonstrate_shap_vs_feature_importance():
    """
    Compare SHAP importance with traditional feature importance.
    """
    print("\n" + "=" * 80)
    print("SHAP VS TRADITIONAL FEATURE IMPORTANCE")
    print("=" * 80)
    
    # Generate data with some correlated features
    np.random.seed(42)
    n_samples = 1000
    
    X1 = np.random.randn(n_samples)
    X2 = X1 + np.random.randn(n_samples) * 0.3  # Correlated with X1
    X3 = np.random.randn(n_samples)
    X4 = np.random.randn(n_samples)
    
    X = np.column_stack([X1, X2, X3, X4])
    y = (X1 + X2 + 2 * X3 + np.random.randn(n_samples) * 0.5 > 0).astype(int)
    
    feature_names = ['X1', 'X2', 'X3', 'X4']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset with Correlated Features:")
    print("   X1 and X2 are correlated (both contribute to prediction)")
    print("   X3 has highest true impact")
    print("   X4 is noise")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Traditional importance
    traditional_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n2. Traditional Feature Importance (Gini/Split-based):")
    print(traditional_importance.to_string(index=False))
    
    # SHAP importance
    analyzer = SHAPAnalyzer(model, X_train)
    analyzer.create_explainer(explainer_type='tree')
    analyzer.explain_predictions(X_test)
    shap_importance = analyzer.get_global_importance(feature_names)
    
    print(f"\n3. SHAP Feature Importance (Mean |SHAP|):")
    print(shap_importance.to_string(index=False))
    
    print(f"\n4. Key Differences:")
    print("   - Traditional importance can be biased by correlations")
    print("   - SHAP properly distributes credit among correlated features")
    print("   - SHAP values are in same units as model output")
    print("   - SHAP provides directional impact (positive/negative)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_shap_basics()
    demonstrate_local_explanations()
    demonstrate_shap_use_cases()
    demonstrate_shap_vs_feature_importance()
    
    print("\n✅ Module 2 Complete: SHAP Explainability")
    print("\nKey Takeaways:")
    print("1. SHAP provides theoretically grounded explanations")
    print("2. Works for both global and local interpretability")
    print("3. Handles feature interactions and correlations")
    print("4. More reliable than traditional feature importance")
    print("5. Essential for regulated industries and high-stakes decisions")
    print("6. Balance between computational cost and explanation quality")
