"""
Day 1 - Module 5: Support Vector Machines and Kernel Methods
Topic: SVMs and the Kernel Trick

This module covers:
- Linear and non-linear SVMs
- Kernel functions (Linear, Polynomial, RBF, Sigmoid)
- Kernel trick for efficient computation
- SVM hyperparameters (C, gamma)
- Multi-class classification with SVMs
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC, SVR
from sklearn.datasets import make_classification, make_circles, make_moons
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')


class SVMExplorer:
    """
    Comprehensive toolkit for exploring Support Vector Machines.
    """
    
    def __init__(self):
        """Initialize the SVM Explorer."""
        self.models = {}
        self.results = {}
    
    def train_linear_svm(self, X_train, y_train, X_test, y_test, C=1.0):
        """
        Train a linear SVM classifier.
        
        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
        C : float, default=1.0
            Regularization parameter
        
        Returns:
        --------
        dict : Results including accuracy and model
        """
        model = SVC(kernel='linear', C=C, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.models['linear'] = model
        self.results['linear'] = {
            'train_score': train_score,
            'test_score': test_score,
            'C': C,
            'n_support_vectors': model.n_support_
        }
        
        return self.results['linear']
    
    def train_polynomial_svm(self, X_train, y_train, X_test, y_test, degree=3, C=1.0):
        """
        Train a polynomial kernel SVM classifier.
        
        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
        degree : int, default=3
            Degree of polynomial kernel
        C : float, default=1.0
            Regularization parameter
        
        Returns:
        --------
        dict : Results including accuracy and model
        """
        model = SVC(kernel='poly', degree=degree, C=C, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.models['polynomial'] = model
        self.results['polynomial'] = {
            'train_score': train_score,
            'test_score': test_score,
            'degree': degree,
            'C': C,
            'n_support_vectors': model.n_support_
        }
        
        return self.results['polynomial']
    
    def train_rbf_svm(self, X_train, y_train, X_test, y_test, C=1.0, gamma='scale'):
        """
        Train an RBF (Radial Basis Function) kernel SVM classifier.
        
        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
        C : float, default=1.0
            Regularization parameter
        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient
        
        Returns:
        --------
        dict : Results including accuracy and model
        """
        model = SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.models['rbf'] = model
        self.results['rbf'] = {
            'train_score': train_score,
            'test_score': test_score,
            'C': C,
            'gamma': gamma,
            'n_support_vectors': model.n_support_
        }
        
        return self.results['rbf']
    
    def train_sigmoid_svm(self, X_train, y_train, X_test, y_test, C=1.0, gamma='scale'):
        """
        Train a sigmoid kernel SVM classifier.
        
        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
        C : float, default=1.0
            Regularization parameter
        gamma : {'scale', 'auto'} or float, default='scale'
            Kernel coefficient
        
        Returns:
        --------
        dict : Results including accuracy and model
        """
        model = SVC(kernel='sigmoid', C=C, gamma=gamma, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        self.models['sigmoid'] = model
        self.results['sigmoid'] = {
            'train_score': train_score,
            'test_score': test_score,
            'C': C,
            'gamma': gamma,
            'n_support_vectors': model.n_support_
        }
        
        return self.results['sigmoid']
    
    def compare_kernels(self, X_train, y_train, X_test, y_test, C=1.0):
        """
        Compare performance of different kernel functions.
        
        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
        C : float, default=1.0
            Regularization parameter
        
        Returns:
        --------
        pd.DataFrame : Comparison of kernel performances
        """
        self.train_linear_svm(X_train, y_train, X_test, y_test, C=C)
        self.train_polynomial_svm(X_train, y_train, X_test, y_test, C=C)
        self.train_rbf_svm(X_train, y_train, X_test, y_test, C=C)
        self.train_sigmoid_svm(X_train, y_train, X_test, y_test, C=C)
        
        comparison = []
        for kernel_name, result in self.results.items():
            comparison.append({
                'Kernel': kernel_name,
                'Train Accuracy': result['train_score'],
                'Test Accuracy': result['test_score'],
                'Support Vectors': result['n_support_vectors'].sum()
            })
        
        return pd.DataFrame(comparison)
    
    def tune_c_parameter(self, X_train, y_train, X_test, y_test, c_values=None):
        """
        Tune the C (regularization) parameter.
        
        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
        c_values : list, optional
            List of C values to try
        
        Returns:
        --------
        pd.DataFrame : Results for different C values
        """
        if c_values is None:
            c_values = [0.001, 0.01, 0.1, 1, 10, 100]
        
        results = []
        for C in c_values:
            model = SVC(kernel='rbf', C=C, random_state=42)
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            results.append({
                'C': C,
                'Train Accuracy': train_score,
                'Test Accuracy': test_score,
                'Support Vectors': model.n_support_.sum()
            })
        
        return pd.DataFrame(results)
    
    def tune_gamma_parameter(self, X_train, y_train, X_test, y_test, gamma_values=None):
        """
        Tune the gamma parameter for RBF kernel.
        
        Parameters:
        -----------
        X_train : np.array
            Training features
        y_train : np.array
            Training labels
        X_test : np.array
            Test features
        y_test : np.array
            Test labels
        gamma_values : list, optional
            List of gamma values to try
        
        Returns:
        --------
        pd.DataFrame : Results for different gamma values
        """
        if gamma_values is None:
            gamma_values = [0.001, 0.01, 0.1, 1, 10]
        
        results = []
        for gamma in gamma_values:
            model = SVC(kernel='rbf', gamma=gamma, random_state=42)
            model.fit(X_train, y_train)
            
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            
            results.append({
                'Gamma': gamma,
                'Train Accuracy': train_score,
                'Test Accuracy': test_score,
                'Support Vectors': model.n_support_.sum()
            })
        
        return pd.DataFrame(results)


class KernelFunctions:
    """
    Custom kernel function implementations for educational purposes.
    """
    
    @staticmethod
    def linear_kernel(X1, X2):
        """
        Linear kernel: K(x, y) = x^T * y
        
        Parameters:
        -----------
        X1 : np.array
            First set of samples
        X2 : np.array
            Second set of samples
        
        Returns:
        --------
        np.array : Kernel matrix
        """
        return np.dot(X1, X2.T)
    
    @staticmethod
    def polynomial_kernel(X1, X2, degree=3, coef0=1):
        """
        Polynomial kernel: K(x, y) = (x^T * y + coef0)^degree
        
        Parameters:
        -----------
        X1 : np.array
            First set of samples
        X2 : np.array
            Second set of samples
        degree : int, default=3
            Degree of polynomial
        coef0 : float, default=1
            Independent term
        
        Returns:
        --------
        np.array : Kernel matrix
        """
        return (np.dot(X1, X2.T) + coef0) ** degree
    
    @staticmethod
    def rbf_kernel(X1, X2, gamma=1.0):
        """
        RBF (Gaussian) kernel: K(x, y) = exp(-gamma * ||x - y||^2)
        
        Parameters:
        -----------
        X1 : np.array
            First set of samples
        X2 : np.array
            Second set of samples
        gamma : float, default=1.0
            Kernel coefficient
        
        Returns:
        --------
        np.array : Kernel matrix
        """
        # Compute squared Euclidean distances
        X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        distances_squared = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        
        return np.exp(-gamma * distances_squared)
    
    @staticmethod
    def sigmoid_kernel(X1, X2, gamma=1.0, coef0=0):
        """
        Sigmoid kernel: K(x, y) = tanh(gamma * x^T * y + coef0)
        
        Parameters:
        -----------
        X1 : np.array
            First set of samples
        X2 : np.array
            Second set of samples
        gamma : float, default=1.0
            Kernel coefficient
        coef0 : float, default=0
            Independent term
        
        Returns:
        --------
        np.array : Kernel matrix
        """
        return np.tanh(gamma * np.dot(X1, X2.T) + coef0)


# Demonstrations

def demonstrate_linear_svm():
    """
    Demonstrate linear SVM on linearly separable data.
    """
    print("=" * 80)
    print("LINEAR SVM DEMONSTRATION")
    print("=" * 80)
    
    # Generate linearly separable data
    X, y = make_classification(
        n_samples=500, 
        n_features=2, 
        n_redundant=0, 
        n_informative=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Test set: {X_test.shape[0]} samples")
    
    # Train linear SVM
    explorer = SVMExplorer()
    results = explorer.train_linear_svm(X_train, y_train, X_test, y_test, C=1.0)
    
    print(f"\n2. Linear SVM Results:")
    print(f"   Train Accuracy: {results['train_score']:.4f}")
    print(f"   Test Accuracy: {results['test_score']:.4f}")
    print(f"   Number of Support Vectors: {results['n_support_vectors'].sum()}")
    
    print("\n" + "=" * 80)


def demonstrate_kernel_comparison():
    """
    Demonstrate different kernel functions on non-linear data.
    """
    print("\n" + "=" * 80)
    print("KERNEL COMPARISON DEMONSTRATION")
    print("=" * 80)
    
    # Generate non-linear data (circles)
    X, y = make_circles(n_samples=500, noise=0.1, factor=0.5, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\n1. Dataset: Non-linear circles data")
    print(f"   {X.shape[0]} samples, {X.shape[1]} features")
    
    # Compare kernels
    explorer = SVMExplorer()
    comparison = explorer.compare_kernels(X_train, y_train, X_test, y_test, C=1.0)
    
    print(f"\n2. Kernel Comparison Results:")
    print(comparison.to_string(index=False))
    
    print("\n3. Analysis:")
    best_kernel = comparison.loc[comparison['Test Accuracy'].idxmax(), 'Kernel']
    print(f"   Best performing kernel: {best_kernel}")
    print(f"   Test accuracy: {comparison['Test Accuracy'].max():.4f}")
    
    print("\n" + "=" * 80)


def demonstrate_hyperparameter_tuning():
    """
    Demonstrate SVM hyperparameter tuning.
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING DEMONSTRATION")
    print("=" * 80)
    
    # Generate data
    X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
    
    # Split and standardize
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\n1. Dataset: Moon-shaped data")
    print(f"   {X.shape[0]} samples, {X.shape[1]} features")
    
    explorer = SVMExplorer()
    
    # Tune C parameter
    print("\n2. Tuning C (Regularization) Parameter:")
    c_results = explorer.tune_c_parameter(X_train, y_train, X_test, y_test)
    print(c_results.to_string(index=False))
    
    best_c = c_results.loc[c_results['Test Accuracy'].idxmax(), 'C']
    print(f"\n   Best C value: {best_c}")
    
    # Tune gamma parameter
    print("\n3. Tuning Gamma Parameter:")
    gamma_results = explorer.tune_gamma_parameter(X_train, y_train, X_test, y_test)
    print(gamma_results.to_string(index=False))
    
    best_gamma = gamma_results.loc[gamma_results['Test Accuracy'].idxmax(), 'Gamma']
    print(f"\n   Best Gamma value: {best_gamma}")
    
    print("\n" + "=" * 80)


def demonstrate_kernel_functions():
    """
    Demonstrate custom kernel function implementations.
    """
    print("\n" + "=" * 80)
    print("KERNEL FUNCTION IMPLEMENTATIONS")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    X1 = np.random.randn(3, 2)
    X2 = np.random.randn(2, 2)
    
    print("\n1. Sample Data:")
    print(f"   X1 shape: {X1.shape}")
    print(f"   X2 shape: {X2.shape}")
    
    # Compute different kernels
    print("\n2. Linear Kernel:")
    linear_K = KernelFunctions.linear_kernel(X1, X2)
    print(f"   Shape: {linear_K.shape}")
    print(f"   Matrix:\n{linear_K}")
    
    print("\n3. Polynomial Kernel (degree=2):")
    poly_K = KernelFunctions.polynomial_kernel(X1, X2, degree=2)
    print(f"   Shape: {poly_K.shape}")
    print(f"   Matrix:\n{poly_K}")
    
    print("\n4. RBF Kernel (gamma=1.0):")
    rbf_K = KernelFunctions.rbf_kernel(X1, X2, gamma=1.0)
    print(f"   Shape: {rbf_K.shape}")
    print(f"   Matrix:\n{rbf_K}")
    
    print("\n5. Sigmoid Kernel:")
    sigmoid_K = KernelFunctions.sigmoid_kernel(X1, X2, gamma=1.0)
    print(f"   Shape: {sigmoid_K.shape}")
    print(f"   Matrix:\n{sigmoid_K}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    demonstrate_linear_svm()
    demonstrate_kernel_comparison()
    demonstrate_hyperparameter_tuning()
    demonstrate_kernel_functions()
    
    print("\n✅ Module 5 Complete: Support Vector Machines and Kernel Methods")
    print("\nKey Takeaways:")
    print("1. Linear SVMs work well for linearly separable data")
    print("2. Kernel functions enable SVMs to learn non-linear decision boundaries")
    print("3. RBF kernel is most commonly used for non-linear problems")
    print("4. C parameter controls the trade-off between margin and misclassification")
    print("5. Gamma parameter controls the influence of individual training samples")
    print("6. The kernel trick allows efficient computation in high-dimensional spaces")
