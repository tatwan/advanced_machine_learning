"""
Day 2 - Module 2: Random Forests
Topic: Understanding and Implementing Random Forests

This module covers:
- Random Forest algorithm theory
- Bagging and bootstrap aggregating
- Feature importance and selection
- Out-of-bag (OOB) error estimation
- Hyperparameter tuning for Random Forests
- Practical applications
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')


class RandomForestAnalyzer:
    """
    Comprehensive Random Forest analysis toolkit.
    """
    
    def __init__(self, task='classification'):
        """
        Initialize the analyzer.
        
        Parameters:
        -----------
        task : str, default='classification'
            Type of task: 'classification' or 'regression'
        """
        self.task = task
        self.model = None
        self.feature_importances_ = None
        self.oob_score_ = None
    
    def train_random_forest(self, X_train, y_train, n_estimators=100, 
                           max_depth=None, min_samples_split=2,
                           max_features='sqrt', oob_score=True):
        """
        Train a Random Forest model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels/values
        n_estimators : int, default=100
            Number of trees
        max_depth : int, optional
            Maximum depth of trees
        min_samples_split : int, default=2
            Minimum samples required to split
        max_features : str or int, default='sqrt'
            Number of features to consider for split
        oob_score : bool, default=True
            Whether to use out-of-bag samples to estimate generalization score
        
        Returns:
        --------
        self : RandomForestAnalyzer
            Fitted analyzer
        """
        if self.task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                oob_score=oob_score,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                max_features=max_features,
                oob_score=oob_score,
                random_state=42,
                n_jobs=-1
            )
        
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_
        
        if oob_score:
            self.oob_score_ = self.model.oob_score_
        
        return self
    
    def get_feature_importance(self, feature_names=None, top_n=10):
        """
        Get feature importance ranking.
        
        Parameters:
        -----------
        feature_names : list, optional
            Names of features
        top_n : int, default=10
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame : Feature importance dataframe
        """
        if self.feature_importances_ is None:
            raise ValueError("Model must be trained first")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importances_))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)
    
    def compare_single_vs_ensemble(self, X_train, y_train, X_test, y_test):
        """
        Compare single decision tree vs Random Forest.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels/values
        X_test : array-like
            Test features
        y_test : array-like
            Test labels/values
        
        Returns:
        --------
        dict : Comparison results
        """
        # Train single decision tree
        if self.task == 'classification':
            single_tree = DecisionTreeClassifier(random_state=42)
            metric_func = accuracy_score
            metric_name = 'accuracy'
        else:
            single_tree = DecisionTreeClassifier(random_state=42)
            metric_func = lambda y_true, y_pred: -mean_squared_error(y_true, y_pred)
            metric_name = 'neg_mse'
        
        single_tree.fit(X_train, y_train)
        single_pred = single_tree.predict(X_test)
        single_score = metric_func(y_test, single_pred)
        
        # Random Forest (already trained)
        rf_pred = self.model.predict(X_test)
        rf_score = metric_func(y_test, rf_pred)
        
        results = {
            'single_tree': {
                metric_name: single_score,
                'estimators': 1
            },
            'random_forest': {
                metric_name: rf_score,
                'estimators': self.model.n_estimators
            },
            'improvement': rf_score - single_score
        }
        
        return results
    
    def analyze_tree_diversity(self):
        """
        Analyze diversity among trees in the forest.
        
        Returns:
        --------
        dict : Diversity statistics
        """
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        # Get predictions from each tree
        tree_predictions = []
        for tree in self.model.estimators_:
            tree_predictions.append(tree.tree_.max_depth)
        
        diversity = {
            'mean_depth': np.mean(tree_predictions),
            'std_depth': np.std(tree_predictions),
            'min_depth': np.min(tree_predictions),
            'max_depth': np.max(tree_predictions)
        }
        
        return diversity
    
    def evaluate_ntrees_effect(self, X_train, y_train, X_test, y_test, 
                               n_estimators_range=[10, 50, 100, 200, 500]):
        """
        Evaluate the effect of number of trees on performance.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels/values
        X_test : array-like
            Test features
        y_test : array-like
            Test labels/values
        n_estimators_range : list, default=[10, 50, 100, 200, 500]
            Different numbers of trees to try
        
        Returns:
        --------
        pd.DataFrame : Results for different n_estimators
        """
        results = []
        
        for n_est in n_estimators_range:
            if self.task == 'classification':
                model = RandomForestClassifier(
                    n_estimators=n_est, random_state=42, n_jobs=-1
                )
                metric_func = accuracy_score
                metric_name = 'accuracy'
            else:
                model = RandomForestRegressor(
                    n_estimators=n_est, random_state=42, n_jobs=-1
                )
                metric_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
                metric_name = 'rmse'
            
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            test_pred = model.predict(X_test)
            
            results.append({
                'n_estimators': n_est,
                f'train_{metric_name}': metric_func(y_train, train_pred),
                f'test_{metric_name}': metric_func(y_test, test_pred)
            })
        
        return pd.DataFrame(results)


class BootstrapAggregating:
    """
    Demonstration of Bootstrap Aggregating (Bagging).
    """
    
    @staticmethod
    def create_bootstrap_sample(X, y, sample_size=None):
        """
        Create a bootstrap sample from the dataset.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Labels/values
        sample_size : int, optional
            Size of bootstrap sample. If None, uses len(X)
        
        Returns:
        --------
        tuple : Bootstrap sample (X_bootstrap, y_bootstrap, oob_indices)
        """
        n_samples = len(X)
        if sample_size is None:
            sample_size = n_samples
        
        # Random sampling with replacement
        bootstrap_indices = np.random.choice(n_samples, size=sample_size, replace=True)
        
        # Out-of-bag indices
        oob_indices = np.array([i for i in range(n_samples) if i not in bootstrap_indices])
        
        return X[bootstrap_indices], y[bootstrap_indices], oob_indices
    
    @staticmethod
    def manual_bagging_classifier(X_train, y_train, X_test, n_estimators=10):
        """
        Manual implementation of bagging for demonstration.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_test : array-like
            Test features
        n_estimators : int, default=10
            Number of base estimators
        
        Returns:
        --------
        np.array : Predictions
        """
        predictions = []
        
        for i in range(n_estimators):
            # Create bootstrap sample
            X_boot, y_boot, _ = BootstrapAggregating.create_bootstrap_sample(
                X_train, y_train
            )
            
            # Train base estimator
            tree = DecisionTreeClassifier(max_depth=10, random_state=i)
            tree.fit(X_boot, y_boot)
            
            # Predict
            pred = tree.predict(X_test)
            predictions.append(pred)
        
        # Aggregate predictions (majority vote)
        predictions = np.array(predictions)
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), 
            axis=0, 
            arr=predictions
        )
        
        return final_predictions


# Demonstrations

def demonstrate_random_forest_basics():
    """
    Demonstrate basic Random Forest functionality.
    """
    print("=" * 80)
    print("RANDOM FOREST BASICS DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train Random Forest
    analyzer = RandomForestAnalyzer(task='classification')
    analyzer.train_random_forest(
        X_train, y_train,
        n_estimators=100,
        max_depth=10,
        oob_score=True
    )
    
    # Evaluate
    train_pred = analyzer.model.predict(X_train)
    test_pred = analyzer.model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n2. Model Performance:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   OOB Score: {analyzer.oob_score_:.4f}")
    
    # Feature importance
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    importance_df = analyzer.get_feature_importance(feature_names, top_n=10)
    
    print(f"\n3. Top 10 Most Important Features:")
    print(importance_df.to_string(index=False))
    
    # Tree diversity
    diversity = analyzer.analyze_tree_diversity()
    print(f"\n4. Tree Diversity Analysis:")
    print(f"   Mean Tree Depth: {diversity['mean_depth']:.2f}")
    print(f"   Std Tree Depth: {diversity['std_depth']:.2f}")
    print(f"   Min Tree Depth: {diversity['min_depth']}")
    print(f"   Max Tree Depth: {diversity['max_depth']}")
    
    print("\n" + "=" * 80)


def demonstrate_single_vs_ensemble():
    """
    Demonstrate single tree vs ensemble comparison.
    """
    print("\n" + "=" * 80)
    print("SINGLE TREE VS RANDOM FOREST COMPARISON")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=800, n_features=15, n_informative=10,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Train and compare
    analyzer = RandomForestAnalyzer(task='classification')
    analyzer.train_random_forest(X_train, y_train, n_estimators=100)
    
    comparison = analyzer.compare_single_vs_ensemble(
        X_train, y_train, X_test, y_test
    )
    
    print(f"\n2. Performance Comparison:")
    print(f"   Single Decision Tree:")
    print(f"      Accuracy: {comparison['single_tree']['accuracy']:.4f}")
    
    print(f"\n   Random Forest (100 trees):")
    print(f"      Accuracy: {comparison['random_forest']['accuracy']:.4f}")
    
    print(f"\n   Improvement: {comparison['improvement']:.4f}")
    
    print("\n" + "=" * 80)


def demonstrate_ntrees_effect():
    """
    Demonstrate the effect of number of trees on performance.
    """
    print("\n" + "=" * 80)
    print("NUMBER OF TREES EFFECT DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Evaluate different numbers of trees
    analyzer = RandomForestAnalyzer(task='classification')
    results = analyzer.evaluate_ntrees_effect(
        X_train, y_train, X_test, y_test,
        n_estimators_range=[10, 50, 100, 200, 500]
    )
    
    print(f"\n2. Performance vs Number of Trees:")
    print(results.to_string(index=False))
    
    print("\n3. Analysis:")
    print("   - Performance improves with more trees initially")
    print("   - Returns diminish after a certain point")
    print("   - More trees = more computation time")
    print("   - Balance between performance and computational cost")
    
    print("\n" + "=" * 80)


def demonstrate_bootstrap_aggregating():
    """
    Demonstrate bootstrap aggregating (bagging) concept.
    """
    print("\n" + "=" * 80)
    print("BOOTSTRAP AGGREGATING (BAGGING) DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=500, n_features=10, n_informative=7,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Training set: {len(X_train)} samples")
    
    # Create bootstrap sample
    X_boot, y_boot, oob_indices = BootstrapAggregating.create_bootstrap_sample(
        X_train, y_train
    )
    
    print(f"\n2. Bootstrap Sample:")
    print(f"   Bootstrap sample size: {len(X_boot)}")
    print(f"   Out-of-bag samples: {len(oob_indices)}")
    print(f"   OOB percentage: {len(oob_indices)/len(X_train)*100:.1f}%")
    
    # Manual bagging
    print(f"\n3. Manual Bagging with 10 estimators:")
    predictions = BootstrapAggregating.manual_bagging_classifier(
        X_train, y_train, X_test, n_estimators=10
    )
    manual_acc = accuracy_score(y_test, predictions)
    print(f"   Accuracy: {manual_acc:.4f}")
    
    # Compare with sklearn RandomForest
    print(f"\n4. Sklearn RandomForest with 10 estimators:")
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"   Accuracy: {rf_acc:.4f}")
    
    print("\n5. Key Concepts:")
    print("   - Bootstrap: Random sampling with replacement")
    print("   - Out-of-bag (OOB): ~37% of samples not selected in each bootstrap")
    print("   - Aggregating: Combining predictions via voting/averaging")
    print("   - Reduces variance and prevents overfitting")
    
    print("\n" + "=" * 80)


def demonstrate_hyperparameter_tuning():
    """
    Demonstrate Random Forest hyperparameter tuning.
    """
    print("\n" + "=" * 80)
    print("RANDOM FOREST HYPERPARAMETER TUNING")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    hyperparameters = [
        {'max_depth': None, 'min_samples_split': 2, 'max_features': 'sqrt'},
        {'max_depth': 10, 'min_samples_split': 2, 'max_features': 'sqrt'},
        {'max_depth': 20, 'min_samples_split': 5, 'max_features': 'sqrt'},
        {'max_depth': None, 'min_samples_split': 2, 'max_features': 'log2'},
    ]
    
    print(f"\n2. Testing Different Hyperparameter Configurations:\n")
    
    results = []
    for i, params in enumerate(hyperparameters, 1):
        rf = RandomForestClassifier(n_estimators=100, random_state=42, **params)
        rf.fit(X_train, y_train)
        
        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)
        
        results.append({
            'config': i,
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'max_features': params['max_features'],
            'train_acc': train_acc,
            'test_acc': test_acc
        })
        
        print(f"   Config {i}: {params}")
        print(f"      Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        print()
    
    results_df = pd.DataFrame(results)
    best_config = results_df.loc[results_df['test_acc'].idxmax()]
    
    print(f"3. Best Configuration (based on test accuracy):")
    print(f"   Config: {int(best_config['config'])}")
    print(f"   Test Accuracy: {best_config['test_acc']:.4f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_random_forest_basics()
    demonstrate_single_vs_ensemble()
    demonstrate_ntrees_effect()
    demonstrate_bootstrap_aggregating()
    demonstrate_hyperparameter_tuning()
    
    print("\n✅ Module 2 Complete: Random Forests")
    print("\nKey Takeaways:")
    print("1. Random Forests combine multiple decision trees via bagging")
    print("2. Bootstrap sampling creates diverse training sets for each tree")
    print("3. Feature randomness adds additional diversity")
    print("4. OOB samples provide unbiased performance estimate")
    print("5. More trees generally improve performance up to a point")
    print("6. Feature importance helps with feature selection and interpretation")
