"""
Day 2 - Module 3: Gradient Boosting Trees
Topic: Gradient Boosting, XGBoost, and LightGBM

This module covers:
- Gradient Boosting algorithm theory
- XGBoost advanced features
- LightGBM for large datasets
- Hyperparameter tuning for boosting
- Comparison of boosting libraries
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')


class GradientBoostingAnalyzer:
    """
    Comprehensive gradient boosting analysis toolkit.
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
        self.models = {}
        self.training_history = {}
    
    def train_sklearn_gb(self, X_train, y_train, n_estimators=100, 
                        learning_rate=0.1, max_depth=3):
        """
        Train sklearn Gradient Boosting model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels/values
        n_estimators : int, default=100
            Number of boosting stages
        learning_rate : float, default=0.1
            Learning rate shrinks contribution of each tree
        max_depth : int, default=3
            Maximum depth of individual trees
        
        Returns:
        --------
        model : Trained model
        """
        if self.task == 'classification':
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        else:
            model = GradientBoostingRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        
        model.fit(X_train, y_train)
        self.models['sklearn_gb'] = model
        self.training_history['sklearn_gb'] = model.train_score_
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val=None, y_val=None,
                     n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Train XGBoost model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels/values
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation labels/values
        n_estimators : int, default=100
            Number of boosting rounds
        learning_rate : float, default=0.1
            Learning rate
        max_depth : int, default=3
            Maximum tree depth
        
        Returns:
        --------
        model : Trained model
        """
        if self.task == 'classification':
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                eval_metric='logloss'
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42
            )
        
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val)]
            model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val=None, y_val=None,
                      n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Train LightGBM model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels/values
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation labels/values
        n_estimators : int, default=100
            Number of boosting rounds
        learning_rate : float, default=0.1
            Learning rate
        max_depth : int, default=3
            Maximum tree depth
        
        Returns:
        --------
        model : Trained model
        """
        if self.task == 'classification':
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                verbose=-1
            )
        else:
            model = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=42,
                verbose=-1
            )
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=10)]
            )
        else:
            model.fit(X_train, y_train)
        
        self.models['lightgbm'] = model
        
        return model
    
    def train_catboost(self, X_train, y_train, X_val=None, y_val=None,
                      n_estimators=100, learning_rate=0.1, max_depth=3):
        """
        Train CatBoost model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels/values
        X_val : array-like, optional
            Validation features
        y_val : array-like, optional
            Validation labels/values
        n_estimators : int, default=100
            Number of boosting rounds
        learning_rate : float, default=0.1
            Learning rate
        max_depth : int, default=3
            Maximum tree depth
        
        Returns:
        --------
        model : Trained model
        """
        if self.task == 'classification':
            model = CatBoostClassifier(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=max_depth,
                random_seed=42,
                verbose=False
            )
        else:
            model = CatBoostRegressor(
                iterations=n_estimators,
                learning_rate=learning_rate,
                depth=max_depth,
                random_seed=42,
                verbose=False
            )
        
        if X_val is not None and y_val is not None:
            model.fit(X_train, y_train, eval_set=(X_val, y_val))
        else:
            model.fit(X_train, y_train)
        
        self.models['catboost'] = model
        
        return model
    
    def compare_all_models(self, X_test, y_test):
        """
        Compare all trained boosting models.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels/values
        
        Returns:
        --------
        pd.DataFrame : Comparison results
        """
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            if self.task == 'classification':
                score = accuracy_score(y_test, y_pred)
                metric_name = 'accuracy'
            else:
                score = np.sqrt(mean_squared_error(y_test, y_pred))
                metric_name = 'rmse'
            
            results.append({
                'model': name,
                metric_name: score
            })
        
        return pd.DataFrame(results)
    
    def get_feature_importance(self, model_name, feature_names=None, top_n=10):
        """
        Get feature importance from a trained model.
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        feature_names : list, optional
            Names of features
        top_n : int, default=10
            Number of top features to return
        
        Returns:
        --------
        pd.DataFrame : Feature importance dataframe
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            raise ValueError(f"Model {model_name} does not have feature_importances_")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importances))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return importance_df.head(top_n)


class LearningRateScheduling:
    """
    Demonstrate learning rate effects in gradient boosting.
    """
    
    @staticmethod
    def compare_learning_rates(X_train, y_train, X_test, y_test, task='classification'):
        """
        Compare different learning rates.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        task : str, default='classification'
            Task type
        
        Returns:
        --------
        pd.DataFrame : Comparison results
        """
        learning_rates = [0.01, 0.05, 0.1, 0.3]
        results = []
        
        for lr in learning_rates:
            if task == 'classification':
                model = GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=lr,
                    max_depth=3,
                    random_state=42
                )
                metric_func = accuracy_score
                metric_name = 'accuracy'
            else:
                model = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=lr,
                    max_depth=3,
                    random_state=42
                )
                metric_func = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))
                metric_name = 'rmse'
            
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            score = metric_func(y_test, test_pred)
            
            results.append({
                'learning_rate': lr,
                metric_name: score
            })
        
        return pd.DataFrame(results)


# Demonstrations

def demonstrate_gradient_boosting_basics():
    """
    Demonstrate basic gradient boosting functionality.
    """
    print("=" * 80)
    print("GRADIENT BOOSTING BASICS DEMONSTRATION")
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
    
    # Train Gradient Boosting
    analyzer = GradientBoostingAnalyzer(task='classification')
    model = analyzer.train_sklearn_gb(X_train, y_train, n_estimators=100)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    print(f"\n2. Gradient Boosting Performance:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    
    # Feature importance
    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    importance_df = analyzer.get_feature_importance('sklearn_gb', feature_names, top_n=10)
    
    print(f"\n3. Top 10 Most Important Features:")
    print(importance_df.to_string(index=False))
    
    print("\n" + "=" * 80)


def demonstrate_library_comparison():
    """
    Compare different gradient boosting libraries.
    """
    print("\n" + "=" * 80)
    print("GRADIENT BOOSTING LIBRARIES COMPARISON")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        random_state=42
    )
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    
    # Initialize analyzer
    analyzer = GradientBoostingAnalyzer(task='classification')
    
    # Train all models
    print(f"\n2. Training Models...")
    print("   - Sklearn Gradient Boosting")
    analyzer.train_sklearn_gb(X_train, y_train, n_estimators=100)
    
    print("   - XGBoost")
    analyzer.train_xgboost(X_train, y_train, X_val, y_val, n_estimators=100)
    
    print("   - LightGBM")
    analyzer.train_lightgbm(X_train, y_train, X_val, y_val, n_estimators=100)
    
    print("   - CatBoost")
    analyzer.train_catboost(X_train, y_train, X_val, y_val, n_estimators=100)
    
    # Compare
    print(f"\n3. Performance Comparison:")
    comparison = analyzer.compare_all_models(X_test, y_test)
    print(comparison.to_string(index=False))
    
    print("\n4. Library Characteristics:")
    print("   Sklearn GB: Traditional implementation, good baseline")
    print("   XGBoost: Industry standard, highly optimized, great performance")
    print("   LightGBM: Faster training, efficient memory usage, handles large datasets")
    print("   CatBoost: Best for categorical features, less tuning required")
    
    print("\n" + "=" * 80)


def demonstrate_learning_rate_effect():
    """
    Demonstrate effect of learning rate on gradient boosting.
    """
    print("\n" + "=" * 80)
    print("LEARNING RATE EFFECT DEMONSTRATION")
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
    
    # Compare learning rates
    results = LearningRateScheduling.compare_learning_rates(
        X_train, y_train, X_test, y_test, task='classification'
    )
    
    print(f"\n2. Learning Rate Comparison:")
    print(results.to_string(index=False))
    
    print("\n3. Analysis:")
    print("   - Lower learning rates: More stable but slower convergence")
    print("   - Higher learning rates: Faster but risk of overshooting")
    print("   - Typically use lower LR with more trees for best results")
    print("   - Balance between learning rate and number of estimators")
    
    print("\n" + "=" * 80)


def demonstrate_early_stopping():
    """
    Demonstrate early stopping in gradient boosting.
    """
    print("\n" + "=" * 80)
    print("EARLY STOPPING DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        random_state=42
    )
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # XGBoost with early stopping
    print(f"\n2. Training XGBoost with Early Stopping...")
    model = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.1,
        random_state=42,
        early_stopping_rounds=10
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    print(f"   Requested estimators: 500")
    print(f"   Best iteration: {model.best_iteration}")
    print(f"   Stopped early: {model.best_iteration < 500}")
    
    # Evaluate
    test_acc = model.score(X_test, y_test)
    print(f"\n3. Test Accuracy: {test_acc:.4f}")
    
    print("\n4. Benefits of Early Stopping:")
    print("   - Prevents overfitting")
    print("   - Saves training time")
    print("   - Automatically finds optimal number of trees")
    print("   - Requires validation set")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_gradient_boosting_basics()
    demonstrate_library_comparison()
    demonstrate_learning_rate_effect()
    demonstrate_early_stopping()
    
    print("\n✅ Module 3 Complete: Gradient Boosting Trees")
    print("\nKey Takeaways:")
    print("1. Gradient Boosting builds trees sequentially, correcting previous errors")
    print("2. XGBoost, LightGBM, and CatBoost offer improved performance")
    print("3. Learning rate controls the contribution of each tree")
    print("4. Early stopping prevents overfitting and saves time")
    print("5. Boosting typically outperforms bagging but is more prone to overfitting")
    print("6. Lower learning rate + more trees often gives best results")
