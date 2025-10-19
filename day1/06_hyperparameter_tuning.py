"""
Day 1 - Module 6: Hyperparameter Tuning
Topic: Advanced Hyperparameter Optimization Strategies

This module covers:
- Grid Search and Random Search
- Bayesian Optimization
- Optuna framework
- Cross-validation strategies
- Hyperparameter importance analysis
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV, 
    RandomizedSearchCV,
    cross_val_score,
    StratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, make_scorer
from scipy.stats import randint, uniform
import optuna
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import warnings
warnings.filterwarnings('ignore')


class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning toolkit.
    """
    
    def __init__(self, model, X_train, y_train, X_test, y_test):
        """
        Initialize the hyperparameter tuner.
        
        Parameters:
        -----------
        model : estimator
            Base model to tune
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = None
        self.best_score = None
        self.tuning_results = {}
    
    def grid_search(self, param_grid, cv=5, scoring='accuracy', n_jobs=-1):
        """
        Perform Grid Search for hyperparameter tuning.
        
        Parameters:
        -----------
        param_grid : dict
            Dictionary with parameters names as keys and lists of parameter settings
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='accuracy'
            Scoring metric
        n_jobs : int, default=-1
            Number of parallel jobs
        
        Returns:
        --------
        dict : Best parameters and scores
        """
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Test on hold-out set
        test_score = grid_search.score(self.X_test, self.y_test)
        
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        results = {
            'method': 'Grid Search',
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'test_score': test_score,
            'total_fits': len(grid_search.cv_results_['params'])
        }
        
        self.tuning_results['grid_search'] = results
        return results
    
    def random_search(self, param_distributions, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1):
        """
        Perform Random Search for hyperparameter tuning.
        
        Parameters:
        -----------
        param_distributions : dict
            Dictionary with parameters names as keys and distributions or lists of parameters
        n_iter : int, default=100
            Number of parameter settings sampled
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='accuracy'
            Scoring metric
        n_jobs : int, default=-1
            Number of parallel jobs
        
        Returns:
        --------
        dict : Best parameters and scores
        """
        random_search = RandomizedSearchCV(
            estimator=self.model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=42
        )
        
        random_search.fit(self.X_train, self.y_train)
        
        # Test on hold-out set
        test_score = random_search.score(self.X_test, self.y_test)
        
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        results = {
            'method': 'Random Search',
            'best_params': random_search.best_params_,
            'best_cv_score': random_search.best_score_,
            'test_score': test_score,
            'total_fits': n_iter
        }
        
        self.tuning_results['random_search'] = results
        return results
    
    def bayesian_optimization(self, search_spaces, n_iter=50, cv=5, scoring='accuracy', n_jobs=-1):
        """
        Perform Bayesian Optimization for hyperparameter tuning.
        
        Parameters:
        -----------
        search_spaces : dict
            Dictionary with parameters names as keys and search space definitions
        n_iter : int, default=50
            Number of parameter settings sampled
        cv : int, default=5
            Number of cross-validation folds
        scoring : str, default='accuracy'
            Scoring metric
        n_jobs : int, default=-1
            Number of parallel jobs
        
        Returns:
        --------
        dict : Best parameters and scores
        """
        bayes_search = BayesSearchCV(
            estimator=self.model,
            search_spaces=search_spaces,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=0,
            random_state=42
        )
        
        bayes_search.fit(self.X_train, self.y_train)
        
        # Test on hold-out set
        test_score = bayes_search.score(self.X_test, self.y_test)
        
        self.best_params = bayes_search.best_params_
        self.best_score = bayes_search.best_score_
        
        results = {
            'method': 'Bayesian Optimization',
            'best_params': bayes_search.best_params_,
            'best_cv_score': bayes_search.best_score_,
            'test_score': test_score,
            'total_fits': n_iter
        }
        
        self.tuning_results['bayesian'] = results
        return results
    
    def optuna_optimization(self, create_trial_params_func, n_trials=100, cv=5):
        """
        Perform hyperparameter tuning using Optuna.
        
        Parameters:
        -----------
        create_trial_params_func : callable
            Function that takes a trial object and returns parameters dict
        n_trials : int, default=100
            Number of trials
        cv : int, default=5
            Number of cross-validation folds
        
        Returns:
        --------
        dict : Best parameters and scores
        """
        def objective(trial):
            params = create_trial_params_func(trial)
            
            # Create model with trial parameters
            model = self.model.__class__(**params)
            
            # Cross-validation score
            scores = cross_val_score(
                model, self.X_train, self.y_train, 
                cv=cv, scoring='accuracy', n_jobs=-1
            )
            
            return scores.mean()
        
        # Create study
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        # Train final model with best parameters
        best_model = self.model.__class__(**study.best_params)
        best_model.fit(self.X_train, self.y_train)
        test_score = best_model.score(self.X_test, self.y_test)
        
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        results = {
            'method': 'Optuna',
            'best_params': study.best_params,
            'best_cv_score': study.best_value,
            'test_score': test_score,
            'total_fits': n_trials
        }
        
        self.tuning_results['optuna'] = results
        return results
    
    def compare_methods(self):
        """
        Compare results from different tuning methods.
        
        Returns:
        --------
        pd.DataFrame : Comparison of tuning methods
        """
        comparison = []
        for method_name, results in self.tuning_results.items():
            comparison.append({
                'Method': results['method'],
                'CV Score': results['best_cv_score'],
                'Test Score': results['test_score'],
                'Total Fits': results['total_fits']
            })
        
        return pd.DataFrame(comparison)


class CrossValidationStrategies:
    """
    Different cross-validation strategies.
    """
    
    @staticmethod
    def standard_kfold(model, X, y, n_splits=5):
        """
        Standard K-Fold cross-validation.
        
        Parameters:
        -----------
        model : estimator
            Model to evaluate
        X : np.array
            Features
        y : np.array
            Labels
        n_splits : int, default=5
            Number of folds
        
        Returns:
        --------
        dict : Cross-validation results
        """
        scores = cross_val_score(model, X, y, cv=n_splits, scoring='accuracy')
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'method': 'Standard K-Fold'
        }
    
    @staticmethod
    def stratified_kfold(model, X, y, n_splits=5):
        """
        Stratified K-Fold cross-validation (maintains class distribution).
        
        Parameters:
        -----------
        model : estimator
            Model to evaluate
        X : np.array
            Features
        y : np.array
            Labels
        n_splits : int, default=5
            Number of folds
        
        Returns:
        --------
        dict : Cross-validation results
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        
        return {
            'scores': scores,
            'mean': scores.mean(),
            'std': scores.std(),
            'method': 'Stratified K-Fold'
        }
    
    @staticmethod
    def nested_cross_validation(model, X, y, param_grid, inner_cv=3, outer_cv=5):
        """
        Nested cross-validation for unbiased performance estimation.
        
        Parameters:
        -----------
        model : estimator
            Model to evaluate
        X : np.array
            Features
        y : np.array
            Labels
        param_grid : dict
            Parameter grid for tuning
        inner_cv : int, default=3
            Inner loop CV folds
        outer_cv : int, default=5
            Outer loop CV folds
        
        Returns:
        --------
        dict : Nested CV results
        """
        outer_scores = []
        
        skf_outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)
        
        for train_idx, test_idx in skf_outer.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Inner loop: hyperparameter tuning
            grid_search = GridSearchCV(
                model, param_grid, cv=inner_cv, scoring='accuracy'
            )
            grid_search.fit(X_train, y_train)
            
            # Outer loop: performance estimation
            score = grid_search.score(X_test, y_test)
            outer_scores.append(score)
        
        return {
            'scores': np.array(outer_scores),
            'mean': np.mean(outer_scores),
            'std': np.std(outer_scores),
            'method': 'Nested CV'
        }


# Demonstrations

def demonstrate_grid_vs_random_search():
    """
    Demonstrate and compare Grid Search vs Random Search.
    """
    print("=" * 80)
    print("GRID SEARCH VS RANDOM SEARCH DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, random_state=42
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10]
    }
    
    # Define parameter distributions for random search
    param_distributions = {
        'n_estimators': randint(50, 200),
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': randint(2, 20)
    }
    
    # Initialize tuner
    rf_model = RandomForestClassifier(random_state=42)
    tuner = HyperparameterTuner(rf_model, X_train, y_train, X_test, y_test)
    
    # Grid Search
    print("\n2. Performing Grid Search...")
    grid_results = tuner.grid_search(param_grid, cv=3)
    print(f"   Best parameters: {grid_results['best_params']}")
    print(f"   Best CV score: {grid_results['best_cv_score']:.4f}")
    print(f"   Test score: {grid_results['test_score']:.4f}")
    print(f"   Total fits: {grid_results['total_fits']}")
    
    # Random Search
    print("\n3. Performing Random Search...")
    random_results = tuner.random_search(param_distributions, n_iter=50, cv=3)
    print(f"   Best parameters: {random_results['best_params']}")
    print(f"   Best CV score: {random_results['best_cv_score']:.4f}")
    print(f"   Test score: {random_results['test_score']:.4f}")
    print(f"   Total fits: {random_results['total_fits']}")
    
    print("\n" + "=" * 80)


def demonstrate_bayesian_optimization():
    """
    Demonstrate Bayesian Optimization for hyperparameter tuning.
    """
    print("\n" + "=" * 80)
    print("BAYESIAN OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=800, n_features=15, n_informative=10,
        random_state=42
    )
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Define search space
    search_spaces = {
        'C': Real(0.01, 100, prior='log-uniform'),
        'gamma': Real(0.001, 1, prior='log-uniform'),
        'kernel': ['rbf', 'poly']
    }
    
    # Initialize tuner
    svm_model = SVC(random_state=42)
    tuner = HyperparameterTuner(svm_model, X_train, y_train, X_test, y_test)
    
    # Bayesian Optimization
    print("\n2. Performing Bayesian Optimization...")
    bayes_results = tuner.bayesian_optimization(search_spaces, n_iter=30, cv=3)
    print(f"   Best parameters: {bayes_results['best_params']}")
    print(f"   Best CV score: {bayes_results['best_cv_score']:.4f}")
    print(f"   Test score: {bayes_results['test_score']:.4f}")
    print(f"   Total fits: {bayes_results['total_fits']}")
    
    print("\n" + "=" * 80)


def demonstrate_optuna():
    """
    Demonstrate Optuna framework for hyperparameter tuning.
    """
    print("\n" + "=" * 80)
    print("OPTUNA FRAMEWORK DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_classification(
        n_samples=800, n_features=15, n_informative=10,
        random_state=42
    )
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Define parameter sampling function
    def create_params(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'random_state': 42
        }
    
    # Initialize tuner
    rf_model = RandomForestClassifier(random_state=42)
    tuner = HyperparameterTuner(rf_model, X_train, y_train, X_test, y_test)
    
    # Optuna optimization
    print("\n2. Performing Optuna Optimization...")
    optuna_results = tuner.optuna_optimization(create_params, n_trials=50, cv=3)
    print(f"   Best parameters: {optuna_results['best_params']}")
    print(f"   Best CV score: {optuna_results['best_cv_score']:.4f}")
    print(f"   Test score: {optuna_results['test_score']:.4f}")
    print(f"   Total fits: {optuna_results['total_fits']}")
    
    print("\n" + "=" * 80)


def demonstrate_cross_validation_strategies():
    """
    Demonstrate different cross-validation strategies.
    """
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION STRATEGIES DEMONSTRATION")
    print("=" * 80)
    
    # Generate imbalanced data
    X, y = make_classification(
        n_samples=1000, n_features=20, n_informative=15,
        n_redundant=5, weights=[0.7, 0.3], random_state=42
    )
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Class distribution: {np.bincount(y)}")
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Standard K-Fold
    print("\n2. Standard K-Fold (5 folds):")
    kfold_results = CrossValidationStrategies.standard_kfold(model, X, y)
    print(f"   Mean accuracy: {kfold_results['mean']:.4f} (+/- {kfold_results['std']:.4f})")
    
    # Stratified K-Fold
    print("\n3. Stratified K-Fold (5 folds):")
    stratified_results = CrossValidationStrategies.stratified_kfold(model, X, y)
    print(f"   Mean accuracy: {stratified_results['mean']:.4f} (+/- {stratified_results['std']:.4f})")
    
    # Nested CV
    print("\n4. Nested Cross-Validation (3x5):")
    param_grid = {'max_depth': [5, 10, 15]}
    nested_results = CrossValidationStrategies.nested_cross_validation(
        model, X, y, param_grid, inner_cv=3, outer_cv=5
    )
    print(f"   Mean accuracy: {nested_results['mean']:.4f} (+/- {nested_results['std']:.4f})")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_grid_vs_random_search()
    demonstrate_bayesian_optimization()
    demonstrate_optuna()
    demonstrate_cross_validation_strategies()
    
    print("\n✅ Module 6 Complete: Hyperparameter Tuning")
    print("\nKey Takeaways:")
    print("1. Grid Search is exhaustive but computationally expensive")
    print("2. Random Search is more efficient for large parameter spaces")
    print("3. Bayesian Optimization is smarter and more efficient")
    print("4. Optuna provides flexible and advanced optimization")
    print("5. Stratified CV maintains class distribution in folds")
    print("6. Nested CV provides unbiased performance estimates")
