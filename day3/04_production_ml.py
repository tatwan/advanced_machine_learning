"""
Day 3 - Module 4: Production ML
Topic: Drift Detection, Monitoring, and Automated Retraining

This module covers:
- Model deployment strategies
- Data drift and concept drift detection
- Model monitoring and logging
- Automated retraining pipelines
- A/B testing in production
- Best practices for ML in production
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DriftDetector:
    """
    Toolkit for detecting data and concept drift.
    """
    
    def __init__(self, reference_data):
        """
        Initialize drift detector.
        
        Parameters:
        -----------
        reference_data : pd.DataFrame or np.array
            Reference (training) data
        """
        self.reference_data = reference_data
        self.reference_stats = self._compute_statistics(reference_data)
    
    def _compute_statistics(self, data):
        """
        Compute statistics for data.
        
        Parameters:
        -----------
        data : array-like
            Data to compute statistics for
        
        Returns:
        --------
        dict : Statistics
        """
        if isinstance(data, pd.DataFrame):
            data = data.values
        
        stats_dict = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'min': np.min(data, axis=0),
            'max': np.max(data, axis=0)
        }
        
        return stats_dict
    
    def detect_data_drift_statistical(self, new_data, alpha=0.05):
        """
        Detect data drift using statistical tests (Kolmogorov-Smirnov test).
        
        Parameters:
        -----------
        new_data : array-like
            New data to compare against reference
        alpha : float, default=0.05
            Significance level
        
        Returns:
        --------
        dict : Drift detection results
        """
        if isinstance(new_data, pd.DataFrame):
            new_data = new_data.values
        if isinstance(self.reference_data, pd.DataFrame):
            reference_data = self.reference_data.values
        else:
            reference_data = self.reference_data
        
        n_features = reference_data.shape[1]
        drift_detected = []
        
        for i in range(n_features):
            # Kolmogorov-Smirnov test
            statistic, p_value = stats.ks_2samp(
                reference_data[:, i],
                new_data[:, i]
            )
            
            drift_detected.append({
                'feature_idx': i,
                'statistic': statistic,
                'p_value': p_value,
                'drift_detected': p_value < alpha
            })
        
        drift_df = pd.DataFrame(drift_detected)
        
        return {
            'overall_drift': drift_df['drift_detected'].any(),
            'n_features_with_drift': drift_df['drift_detected'].sum(),
            'drift_by_feature': drift_df
        }
    
    def detect_data_drift_psi(self, new_data, n_bins=10, threshold=0.1):
        """
        Detect data drift using Population Stability Index (PSI).
        
        Parameters:
        -----------
        new_data : array-like
            New data to compare
        n_bins : int, default=10
            Number of bins for discretization
        threshold : float, default=0.1
            PSI threshold (0.1 = small drift, 0.25 = significant)
        
        Returns:
        --------
        dict : PSI results
        """
        if isinstance(new_data, pd.DataFrame):
            new_data = new_data.values
        if isinstance(self.reference_data, pd.DataFrame):
            reference_data = self.reference_data.values
        else:
            reference_data = self.reference_data
        
        n_features = reference_data.shape[1]
        psi_values = []
        
        for i in range(n_features):
            # Create bins based on reference data
            _, bin_edges = np.histogram(reference_data[:, i], bins=n_bins)
            
            # Bin both datasets
            ref_binned = np.histogram(reference_data[:, i], bins=bin_edges)[0]
            new_binned = np.histogram(new_data[:, i], bins=bin_edges)[0]
            
            # Convert to percentages
            ref_pct = ref_binned / len(reference_data)
            new_pct = new_binned / len(new_data)
            
            # Avoid division by zero
            ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
            new_pct = np.where(new_pct == 0, 0.0001, new_pct)
            
            # Calculate PSI
            psi = np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct))
            
            psi_values.append({
                'feature_idx': i,
                'psi': psi,
                'drift_detected': psi > threshold
            })
        
        psi_df = pd.DataFrame(psi_values)
        
        return {
            'overall_drift': psi_df['drift_detected'].any(),
            'mean_psi': psi_df['psi'].mean(),
            'max_psi': psi_df['psi'].max(),
            'psi_by_feature': psi_df
        }


class ConceptDriftDetector:
    """
    Detect concept drift (changes in target distribution).
    """
    
    @staticmethod
    def detect_accuracy_drift(predictions_history, window_size=100, threshold=0.05):
        """
        Detect drift based on rolling accuracy changes.
        
        Parameters:
        -----------
        predictions_history : list of tuples
            List of (y_true, y_pred) tuples
        window_size : int, default=100
            Size of sliding window
        threshold : float, default=0.05
            Accuracy drop threshold
        
        Returns:
        --------
        dict : Drift detection results
        """
        if len(predictions_history) < window_size * 2:
            return {'drift_detected': False, 'message': 'Insufficient data'}
        
        # Calculate rolling accuracy
        accuracies = []
        for i in range(len(predictions_history) - window_size + 1):
            window = predictions_history[i:i+window_size]
            y_true = [item[0] for item in window]
            y_pred = [item[1] for item in window]
            acc = accuracy_score(y_true, y_pred)
            accuracies.append(acc)
        
        # Compare recent vs baseline
        baseline_acc = np.mean(accuracies[:window_size])
        recent_acc = np.mean(accuracies[-window_size:])
        
        drift_detected = (baseline_acc - recent_acc) > threshold
        
        return {
            'drift_detected': drift_detected,
            'baseline_accuracy': baseline_acc,
            'recent_accuracy': recent_acc,
            'accuracy_drop': baseline_acc - recent_acc,
            'accuracy_history': accuracies
        }
    
    @staticmethod
    def detect_prediction_drift(predictions, reference_predictions, alpha=0.05):
        """
        Detect drift in prediction distributions.
        
        Parameters:
        -----------
        predictions : array-like
            Current predictions
        reference_predictions : array-like
            Reference predictions
        alpha : float, default=0.05
            Significance level
        
        Returns:
        --------
        dict : Drift detection results
        """
        # For classification: compare distribution of predictions
        if len(np.unique(predictions)) < 20:
            # Categorical - use Chi-square test
            ref_counts = np.bincount(reference_predictions.astype(int))
            new_counts = np.bincount(predictions.astype(int))
            
            # Make same length
            max_len = max(len(ref_counts), len(new_counts))
            ref_counts = np.pad(ref_counts, (0, max_len - len(ref_counts)))
            new_counts = np.pad(new_counts, (0, max_len - len(new_counts)))
            
            statistic, p_value = stats.chisquare(new_counts, ref_counts)
        else:
            # Continuous - use KS test
            statistic, p_value = stats.ks_2samp(reference_predictions, predictions)
        
        return {
            'drift_detected': p_value < alpha,
            'test_statistic': statistic,
            'p_value': p_value
        }


class ModelMonitor:
    """
    Comprehensive model monitoring toolkit.
    """
    
    def __init__(self):
        """Initialize model monitor."""
        self.metrics_history = []
        self.alerts = []
    
    def log_prediction(self, y_true, y_pred, y_pred_proba=None, features=None, timestamp=None):
        """
        Log a prediction for monitoring.
        
        Parameters:
        -----------
        y_true : int
            True label
        y_pred : int
            Predicted label
        y_pred_proba : array-like, optional
            Prediction probabilities
        features : array-like, optional
            Input features
        timestamp : datetime, optional
            Prediction timestamp
        """
        log_entry = {
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'features': features,
            'timestamp': timestamp if timestamp else pd.Timestamp.now()
        }
        
        self.metrics_history.append(log_entry)
    
    def get_performance_metrics(self, window_size=None):
        """
        Calculate performance metrics.
        
        Parameters:
        -----------
        window_size : int, optional
            Number of recent predictions to consider
        
        Returns:
        --------
        dict : Performance metrics
        """
        if not self.metrics_history:
            return {}
        
        history = self.metrics_history[-window_size:] if window_size else self.metrics_history
        
        y_true = [entry['y_true'] for entry in history if entry['y_true'] is not None]
        y_pred = [entry['y_pred'] for entry in history if entry['y_true'] is not None]
        
        if not y_true:
            return {'message': 'No labeled data available'}
        
        accuracy = accuracy_score(y_true, y_pred)
        
        return {
            'accuracy': accuracy,
            'n_predictions': len(history),
            'n_labeled': len(y_true)
        }
    
    def check_alerts(self, accuracy_threshold=0.80, response_time_threshold=1.0):
        """
        Check for alert conditions.
        
        Parameters:
        -----------
        accuracy_threshold : float, default=0.80
            Minimum acceptable accuracy
        response_time_threshold : float, default=1.0
            Maximum acceptable response time (seconds)
        
        Returns:
        --------
        list : List of alerts
        """
        current_alerts = []
        
        # Check accuracy
        metrics = self.get_performance_metrics(window_size=100)
        if 'accuracy' in metrics and metrics['accuracy'] < accuracy_threshold:
            current_alerts.append({
                'type': 'accuracy_degradation',
                'message': f"Accuracy {metrics['accuracy']:.3f} below threshold {accuracy_threshold}",
                'severity': 'high'
            })
        
        self.alerts.extend(current_alerts)
        return current_alerts


class RetrainingPipeline:
    """
    Automated model retraining pipeline.
    """
    
    def __init__(self, model, retraining_threshold=0.05):
        """
        Initialize retraining pipeline.
        
        Parameters:
        -----------
        model : estimator
            Base model to retrain
        retraining_threshold : float, default=0.05
            Performance drop threshold to trigger retraining
        """
        self.base_model = model
        self.current_model = model
        self.retraining_threshold = retraining_threshold
        self.retraining_history = []
        self.baseline_performance = None
    
    def should_retrain(self, current_performance):
        """
        Decide if model should be retrained.
        
        Parameters:
        -----------
        current_performance : float
            Current model performance
        
        Returns:
        --------
        bool : Whether to retrain
        """
        if self.baseline_performance is None:
            return False
        
        performance_drop = self.baseline_performance - current_performance
        return performance_drop > self.retraining_threshold
    
    def retrain(self, X_train, y_train, X_val, y_val):
        """
        Retrain the model.
        
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
        
        Returns:
        --------
        dict : Retraining results
        """
        # Clone and train new model
        from sklearn.base import clone
        new_model = clone(self.base_model)
        new_model.fit(X_train, y_train)
        
        # Evaluate new model
        new_performance = new_model.score(X_val, y_val)
        
        # Compare with current model
        if hasattr(self.current_model, 'score'):
            current_performance = self.current_model.score(X_val, y_val)
        else:
            current_performance = 0
        
        # Update if better
        if new_performance >= current_performance:
            self.current_model = new_model
            self.baseline_performance = new_performance
            updated = True
        else:
            updated = False
        
        # Log retraining
        self.retraining_history.append({
            'timestamp': pd.Timestamp.now(),
            'new_performance': new_performance,
            'old_performance': current_performance,
            'model_updated': updated
        })
        
        return {
            'new_performance': new_performance,
            'old_performance': current_performance,
            'model_updated': updated,
            'improvement': new_performance - current_performance
        }


# Demonstrations

def demonstrate_data_drift_detection():
    """
    Demonstrate data drift detection.
    """
    print("=" * 80)
    print("DATA DRIFT DETECTION DEMONSTRATION")
    print("=" * 80)
    
    # Create reference data
    np.random.seed(42)
    reference_data = np.random.randn(1000, 5)
    
    # Create new data with drift in feature 1 and 3
    new_data_no_drift = np.random.randn(500, 5)
    new_data_with_drift = np.random.randn(500, 5)
    new_data_with_drift[:, 1] += 1.5  # Shift feature 1
    new_data_with_drift[:, 3] *= 2.0  # Scale feature 3
    
    print(f"\n1. Reference Data: {reference_data.shape[0]} samples, {reference_data.shape[1]} features")
    
    # Initialize detector
    detector = DriftDetector(reference_data)
    
    # Test without drift
    print(f"\n2. Testing New Data (No Drift):")
    result_no_drift = detector.detect_data_drift_statistical(new_data_no_drift)
    print(f"   Overall Drift Detected: {result_no_drift['overall_drift']}")
    print(f"   Features with Drift: {result_no_drift['n_features_with_drift']}")
    
    # Test with drift
    print(f"\n3. Testing New Data (With Drift in Features 1 and 3):")
    result_with_drift = detector.detect_data_drift_statistical(new_data_with_drift)
    print(f"   Overall Drift Detected: {result_with_drift['overall_drift']}")
    print(f"   Features with Drift: {result_with_drift['n_features_with_drift']}")
    
    drifted_features = result_with_drift['drift_by_feature'][
        result_with_drift['drift_by_feature']['drift_detected']
    ]
    print(f"\n   Drifted Features:")
    print(drifted_features[['feature_idx', 'p_value']].to_string(index=False))
    
    # PSI test
    print(f"\n4. Population Stability Index (PSI) Test:")
    psi_result = detector.detect_data_drift_psi(new_data_with_drift)
    print(f"   Mean PSI: {psi_result['mean_psi']:.4f}")
    print(f"   Max PSI: {psi_result['max_psi']:.4f}")
    print(f"   Overall Drift: {psi_result['overall_drift']}")
    
    print("\n" + "=" * 80)


def demonstrate_concept_drift():
    """
    Demonstrate concept drift detection.
    """
    print("\n" + "=" * 80)
    print("CONCEPT DRIFT DETECTION DEMONSTRATION")
    print("=" * 80)
    
    # Simulate predictions over time with gradual accuracy degradation
    np.random.seed(42)
    n_predictions = 1000
    
    predictions_history = []
    base_accuracy = 0.90
    
    for i in range(n_predictions):
        # Simulate gradual accuracy drop
        current_accuracy = base_accuracy - (i / n_predictions) * 0.15
        
        # Generate prediction
        y_true = np.random.randint(0, 2)
        y_pred = y_true if np.random.random() < current_accuracy else 1 - y_true
        
        predictions_history.append((y_true, y_pred))
    
    print(f"\n1. Simulated {n_predictions} predictions with gradual accuracy drop")
    print(f"   Starting accuracy: {base_accuracy:.2f}")
    print(f"   Ending accuracy: ~{base_accuracy - 0.15:.2f}")
    
    # Detect drift
    result = ConceptDriftDetector.detect_accuracy_drift(
        predictions_history, window_size=100, threshold=0.05
    )
    
    print(f"\n2. Concept Drift Detection Results:")
    print(f"   Drift Detected: {result['drift_detected']}")
    print(f"   Baseline Accuracy: {result['baseline_accuracy']:.4f}")
    print(f"   Recent Accuracy: {result['recent_accuracy']:.4f}")
    print(f"   Accuracy Drop: {result['accuracy_drop']:.4f}")
    
    print("\n" + "=" * 80)


def demonstrate_model_monitoring():
    """
    Demonstrate model monitoring.
    """
    print("\n" + "=" * 80)
    print("MODEL MONITORING DEMONSTRATION")
    print("=" * 80)
    
    # Initialize monitor
    monitor = ModelMonitor()
    
    # Simulate predictions
    np.random.seed(42)
    n_predictions = 200
    
    print(f"\n1. Simulating {n_predictions} predictions...")
    
    for i in range(n_predictions):
        y_true = np.random.randint(0, 2)
        # Simulate degrading performance
        accuracy = 0.90 - (i / n_predictions) * 0.20
        y_pred = y_true if np.random.random() < accuracy else 1 - y_true
        
        monitor.log_prediction(y_true, y_pred)
    
    # Get metrics
    print(f"\n2. Overall Performance:")
    metrics = monitor.get_performance_metrics()
    print(f"   Accuracy: {metrics['accuracy']:.4f}")
    print(f"   Total Predictions: {metrics['n_predictions']}")
    
    print(f"\n3. Recent Performance (last 50 predictions):")
    recent_metrics = monitor.get_performance_metrics(window_size=50)
    print(f"   Accuracy: {recent_metrics['accuracy']:.4f}")
    
    # Check alerts
    print(f"\n4. Alert Check:")
    alerts = monitor.check_alerts(accuracy_threshold=0.80)
    if alerts:
        for alert in alerts:
            print(f"   [{alert['severity'].upper()}] {alert['message']}")
    else:
        print("   No alerts triggered")
    
    print("\n" + "=" * 80)


def demonstrate_automated_retraining():
    """
    Demonstrate automated retraining pipeline.
    """
    print("\n" + "=" * 80)
    print("AUTOMATED RETRAINING PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    # Generate initial training data
    np.random.seed(42)
    X_train, y_train = make_classification(
        n_samples=500, n_features=10, n_informative=7, random_state=42
    )
    X_val, y_val = make_classification(
        n_samples=200, n_features=10, n_informative=7, random_state=43
    )
    
    # Train initial model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    
    initial_performance = model.score(X_val, y_val)
    
    print(f"\n1. Initial Model Performance: {initial_performance:.4f}")
    
    # Initialize pipeline
    pipeline = RetrainingPipeline(model, retraining_threshold=0.03)
    pipeline.baseline_performance = initial_performance
    
    # Simulate performance degradation
    print(f"\n2. Monitoring Performance...")
    
    for i in range(5):
        # Simulate degraded performance
        degraded_performance = initial_performance - (i * 0.02)
        
        should_retrain = pipeline.should_retrain(degraded_performance)
        
        print(f"\n   Iteration {i+1}:")
        print(f"      Current Performance: {degraded_performance:.4f}")
        print(f"      Should Retrain: {should_retrain}")
        
        if should_retrain:
            # Generate new training data (simulating new data collection)
            X_new, y_new = make_classification(
                n_samples=600, n_features=10, n_informative=7, random_state=42+i
            )
            
            result = pipeline.retrain(X_new, y_new, X_val, y_val)
            
            print(f"      Retraining Result:")
            print(f"         New Performance: {result['new_performance']:.4f}")
            print(f"         Improvement: {result['improvement']:.4f}")
            print(f"         Model Updated: {result['model_updated']}")
    
    print(f"\n3. Retraining History: {len(pipeline.retraining_history)} retraining events")
    
    print("\n" + "=" * 80)


def demonstrate_production_best_practices():
    """
    Summarize production ML best practices.
    """
    print("\n" + "=" * 80)
    print("PRODUCTION ML BEST PRACTICES")
    print("=" * 80)
    
    best_practices = {
        "Monitoring": [
            "Log all predictions with timestamps",
            "Track performance metrics continuously",
            "Set up alerts for performance degradation",
            "Monitor data drift and concept drift",
            "Track model latency and throughput"
        ],
        "Versioning": [
            "Version control models and code",
            "Track model lineage and dependencies",
            "Maintain reproducible training pipelines",
            "Document model changes and performance",
            "Keep model metadata (hyperparameters, metrics)"
        ],
        "Testing": [
            "Implement canary deployments",
            "Use A/B testing for model comparison",
            "Shadow mode testing before full deployment",
            "Validate on holdout sets regularly",
            "Test edge cases and failure modes"
        ],
        "Retraining": [
            "Automate retraining pipelines",
            "Set clear retraining triggers",
            "Validate new models before deployment",
            "Maintain rollback capabilities",
            "Balance frequency vs cost"
        ],
        "Infrastructure": [
            "Separate training and serving infrastructure",
            "Implement feature stores for consistency",
            "Use model serving frameworks",
            "Cache predictions when appropriate",
            "Plan for scaling and failover"
        ]
    }
    
    print("\n")
    for category, practices in best_practices.items():
        print(f"{category}:")
        for practice in practices:
            print(f"   • {practice}")
        print()
    
    print("Key Principles:")
    print("   1. Treat models as code - version, test, review")
    print("   2. Monitor everything - data, predictions, performance")
    print("   3. Automate where possible - reduce manual intervention")
    print("   4. Plan for failure - rollback, alerts, fallbacks")
    print("   5. Iterate based on production feedback")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_data_drift_detection()
    demonstrate_concept_drift()
    demonstrate_model_monitoring()
    demonstrate_automated_retraining()
    demonstrate_production_best_practices()
    
    print("\n✅ Module 4 Complete: Production ML")
    print("\nKey Takeaways:")
    print("1. Monitor both data drift and concept drift")
    print("2. Implement automated alerting for performance issues")
    print("3. Build retraining pipelines with clear triggers")
    print("4. Test new models thoroughly before deployment")
    print("5. Maintain model versioning and lineage")
    print("6. Plan for scaling, monitoring, and failure scenarios")
