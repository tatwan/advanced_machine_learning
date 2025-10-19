"""
Day 2 - Module 5: Probabilistic Approaches
Topic: Bayesian Methods and Uncertainty Quantification

This module covers:
- Bayesian inference basics
- Bayesian optimization for hyperparameter tuning
- Probabilistic classifiers
- Uncertainty quantification
- Practical applications
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')


class BayesianInference:
    """
    Bayesian inference demonstrations.
    """
    
    @staticmethod
    def bayesian_coin_flip(n_flips, n_heads):
        """
        Bayesian inference for coin flip probability.
        
        Parameters:
        -----------
        n_flips : int
            Number of coin flips
        n_heads : int
            Number of heads observed
        
        Returns:
        --------
        dict : Posterior statistics
        """
        # Prior: Uniform (Beta(1, 1))
        alpha_prior = 1
        beta_prior = 1
        
        # Posterior: Beta(alpha_prior + n_heads, beta_prior + n_tails)
        alpha_post = alpha_prior + n_heads
        beta_post = beta_prior + (n_flips - n_heads)
        
        # Posterior statistics
        post_mean = alpha_post / (alpha_post + beta_post)
        post_mode = (alpha_post - 1) / (alpha_post + beta_post - 2) if alpha_post > 1 and beta_post > 1 else None
        post_std = np.sqrt((alpha_post * beta_post) / ((alpha_post + beta_post)**2 * (alpha_post + beta_post + 1)))
        
        # Credible interval (95%)
        credible_interval = stats.beta.interval(0.95, alpha_post, beta_post)
        
        return {
            'posterior_mean': post_mean,
            'posterior_mode': post_mode,
            'posterior_std': post_std,
            'credible_interval_95': credible_interval,
            'alpha': alpha_post,
            'beta': beta_post
        }
    
    @staticmethod
    def bayesian_ab_test(conversions_a, trials_a, conversions_b, trials_b, n_samples=10000):
        """
        Bayesian A/B test.
        
        Parameters:
        -----------
        conversions_a : int
            Number of conversions in variant A
        trials_a : int
            Number of trials in variant A
        conversions_b : int
            Number of conversions in variant B
        trials_b : int
            Number of trials in variant B
        n_samples : int, default=10000
            Number of samples for Monte Carlo simulation
        
        Returns:
        --------
        dict : A/B test results
        """
        # Posterior distributions
        alpha_a = 1 + conversions_a
        beta_a = 1 + trials_a - conversions_a
        
        alpha_b = 1 + conversions_b
        beta_b = 1 + trials_b - conversions_b
        
        # Sample from posteriors
        samples_a = np.random.beta(alpha_a, beta_a, n_samples)
        samples_b = np.random.beta(alpha_b, beta_b, n_samples)
        
        # Probability that B is better than A
        prob_b_better = (samples_b > samples_a).mean()
        
        # Expected lift
        lift_samples = (samples_b - samples_a) / samples_a
        expected_lift = lift_samples.mean()
        lift_credible_interval = np.percentile(lift_samples, [2.5, 97.5])
        
        return {
            'prob_b_better_than_a': prob_b_better,
            'expected_lift': expected_lift,
            'lift_credible_interval': lift_credible_interval,
            'conversion_rate_a': conversions_a / trials_a,
            'conversion_rate_b': conversions_b / trials_b
        }


class BayesianOptimization:
    """
    Bayesian optimization for hyperparameter tuning.
    """
    
    @staticmethod
    def optimize_model(X_train, y_train, X_val, y_val, n_calls=30):
        """
        Use Bayesian optimization to tune model hyperparameters.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_val : array-like
            Validation features
        y_val : array-like
            Validation labels
        n_calls : int, default=30
            Number of optimization iterations
        
        Returns:
        --------
        dict : Optimization results
        """
        from sklearn.ensemble import RandomForestClassifier
        
        # Define search space
        space = [
            Integer(10, 200, name='n_estimators'),
            Integer(2, 20, name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 10, name='min_samples_leaf')
        ]
        
        # Define objective function
        @use_named_args(space)
        def objective(**params):
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            
            # Return negative score because gp_minimize minimizes
            return -score
        
        # Perform optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=False
        )
        
        # Best parameters
        best_params = {
            'n_estimators': result.x[0],
            'max_depth': result.x[1],
            'min_samples_split': result.x[2],
            'min_samples_leaf': result.x[3]
        }
        
        return {
            'best_params': best_params,
            'best_score': -result.fun,  # Convert back to positive
            'n_iterations': len(result.func_vals)
        }


class UncertaintyQuantification:
    """
    Methods for quantifying prediction uncertainty.
    """
    
    @staticmethod
    def prediction_intervals_bootstrap(model, X_train, y_train, X_test, n_iterations=100, confidence=0.95):
        """
        Compute prediction intervals using bootstrap.
        
        Parameters:
        -----------
        model : estimator
            Base model
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_test : array-like
            Test features
        n_iterations : int, default=100
            Number of bootstrap iterations
        confidence : float, default=0.95
            Confidence level
        
        Returns:
        --------
        dict : Predictions with intervals
        """
        predictions = []
        
        n_samples = len(X_train)
        
        for i in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train and predict
            model_boot = model.__class__(**model.get_params())
            model_boot.fit(X_boot, y_boot)
            pred = model_boot.predict(X_test)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        # Compute intervals
        lower_percentile = ((1 - confidence) / 2) * 100
        upper_percentile = (confidence + (1 - confidence) / 2) * 100
        
        mean_pred = predictions.mean(axis=0)
        lower_bound = np.percentile(predictions, lower_percentile, axis=0)
        upper_bound = np.percentile(predictions, upper_percentile, axis=0)
        
        return {
            'mean_prediction': mean_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': predictions.std(axis=0)
        }
    
    @staticmethod
    def calibration_assessment(y_true, y_pred_proba, n_bins=10):
        """
        Assess calibration of probability predictions.
        
        Parameters:
        -----------
        y_true : array-like
            True labels
        y_pred_proba : array-like
            Predicted probabilities
        n_bins : int, default=10
            Number of bins for calibration curve
        
        Returns:
        --------
        dict : Calibration metrics
        """
        # Create bins
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_pred_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        
        # Compute calibration per bin
        calibration_data = []
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                mean_predicted = y_pred_proba[mask].mean()
                mean_observed = y_true[mask].mean()
                count = mask.sum()
                
                calibration_data.append({
                    'bin': i,
                    'mean_predicted': mean_predicted,
                    'mean_observed': mean_observed,
                    'count': count
                })
        
        calibration_df = pd.DataFrame(calibration_data)
        
        # Expected Calibration Error (ECE)
        ece = np.abs(calibration_df['mean_predicted'] - calibration_df['mean_observed']).mean()
        
        return {
            'calibration_curve': calibration_df,
            'expected_calibration_error': ece
        }


# Demonstrations

def demonstrate_bayesian_inference():
    """
    Demonstrate Bayesian inference concepts.
    """
    print("=" * 80)
    print("BAYESIAN INFERENCE DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. Example: Coin Flip Inference")
    print("   Observed: 60 heads out of 100 flips")
    
    result = BayesianInference.bayesian_coin_flip(100, 60)
    
    print(f"\n2. Posterior Distribution:")
    print(f"   Mean: {result['posterior_mean']:.4f}")
    print(f"   Std: {result['posterior_std']:.4f}")
    print(f"   95% Credible Interval: [{result['credible_interval_95'][0]:.4f}, {result['credible_interval_95'][1]:.4f}]")
    
    print("\n3. Interpretation:")
    print("   - We're 95% confident the true probability is in the credible interval")
    print("   - Unlike frequentist CI, this is a probability statement about the parameter")
    print("   - Incorporates both prior beliefs and observed data")
    
    print("\n" + "=" * 80)


def demonstrate_bayesian_ab_test():
    """
    Demonstrate Bayesian A/B testing.
    """
    print("\n" + "=" * 80)
    print("BAYESIAN A/B TESTING DEMONSTRATION")
    print("=" * 80)
    
    print("\n1. Scenario: Website Conversion Rate Test")
    print("   Variant A: 120 conversions out of 1000 visitors")
    print("   Variant B: 145 conversions out of 1000 visitors")
    
    result = BayesianInference.bayesian_ab_test(120, 1000, 145, 1000)
    
    print(f"\n2. Results:")
    print(f"   Conversion Rate A: {result['conversion_rate_a']:.2%}")
    print(f"   Conversion Rate B: {result['conversion_rate_b']:.2%}")
    print(f"   Probability B > A: {result['prob_b_better_than_a']:.2%}")
    print(f"   Expected Lift: {result['expected_lift']:.2%}")
    print(f"   Lift 95% Credible Interval: [{result['lift_credible_interval'][0]:.2%}, {result['lift_credible_interval'][1]:.2%}]")
    
    print("\n3. Decision:")
    if result['prob_b_better_than_a'] > 0.95:
        print("   ✓ Strong evidence for B (>95% probability)")
    elif result['prob_b_better_than_a'] > 0.90:
        print("   ~ Moderate evidence for B (>90% probability)")
    else:
        print("   ✗ Insufficient evidence to declare a winner")
    
    print("\n" + "=" * 80)


def demonstrate_bayesian_optimization():
    """
    Demonstrate Bayesian optimization for hyperparameter tuning.
    """
    print("\n" + "=" * 80)
    print("BAYESIAN OPTIMIZATION DEMONSTRATION")
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
    
    print(f"\n2. Performing Bayesian Optimization (30 iterations)...")
    
    result = BayesianOptimization.optimize_model(
        X_train, y_train, X_val, y_val, n_calls=30
    )
    
    print(f"\n3. Optimization Results:")
    print(f"   Best Parameters:")
    for param, value in result['best_params'].items():
        print(f"      {param}: {value}")
    print(f"   Best Validation Score: {result['best_score']:.4f}")
    
    print("\n4. Advantages of Bayesian Optimization:")
    print("   - More efficient than grid/random search")
    print("   - Uses past evaluations to inform future searches")
    print("   - Balances exploration and exploitation")
    print("   - Great for expensive objective functions")
    
    print("\n" + "=" * 80)


def demonstrate_uncertainty_quantification():
    """
    Demonstrate uncertainty quantification methods.
    """
    print("\n" + "=" * 80)
    print("UNCERTAINTY QUANTIFICATION DEMONSTRATION")
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
    
    # Train model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    # Prediction with uncertainty
    print(f"\n2. Computing Prediction Intervals (Bootstrap)...")
    intervals = UncertaintyQuantification.prediction_intervals_bootstrap(
        model, X_train, y_train, X_test[:10], n_iterations=50
    )
    
    print(f"\n3. Sample Predictions with Uncertainty:")
    for i in range(min(5, len(X_test))):
        print(f"   Sample {i+1}:")
        print(f"      Mean Prediction: {intervals['mean_prediction'][i]:.2f}")
        print(f"      Std: {intervals['std'][i]:.2f}")
        print(f"      95% Interval: [{intervals['lower_bound'][i]:.2f}, {intervals['upper_bound'][i]:.2f}]")
    
    # Calibration
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    calibration = UncertaintyQuantification.calibration_assessment(y_test, y_pred_proba)
    
    print(f"\n4. Calibration Assessment:")
    print(f"   Expected Calibration Error: {calibration['expected_calibration_error']:.4f}")
    print(f"   (Lower is better; 0 = perfectly calibrated)")
    
    print("\n5. Importance of Uncertainty:")
    print("   - Critical for high-stakes decisions")
    print("   - Helps identify when model is confident vs uncertain")
    print("   - Enables risk-aware decision making")
    print("   - Important for model deployment and monitoring")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_bayesian_inference()
    demonstrate_bayesian_ab_test()
    demonstrate_bayesian_optimization()
    demonstrate_uncertainty_quantification()
    
    print("\n✅ Module 5 Complete: Probabilistic Approaches")
    print("\nKey Takeaways:")
    print("1. Bayesian methods provide probabilistic interpretations")
    print("2. Credible intervals are probability statements about parameters")
    print("3. Bayesian optimization is efficient for hyperparameter tuning")
    print("4. Uncertainty quantification is crucial for reliable predictions")
    print("5. Calibration ensures predicted probabilities are meaningful")
    print("6. Bootstrap provides practical uncertainty estimates")
