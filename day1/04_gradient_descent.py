"""
Day 1 - Module 4: Gradient Descent
Topic: Gradient Descent Variants and Optimization

This module covers:
- Batch, Mini-batch, and Stochastic Gradient Descent
- Momentum and Nesterov Accelerated Gradient
- Adaptive learning rate methods (AdaGrad, RMSprop, Adam)
- Learning rate scheduling
- Implementation from scratch
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_regression
import warnings
warnings.filterwarnings('ignore')


class GradientDescentOptimizer:
    """
    Implementation of various gradient descent optimization algorithms.
    """
    
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """
        Initialize the optimizer.
        
        Parameters:
        -----------
        learning_rate : float, default=0.01
            Learning rate for gradient descent
        max_iterations : int, default=1000
            Maximum number of iterations
        tolerance : float, default=1e-6
            Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.loss_history = []
        self.weights = None
        self.bias = None
    
    def _compute_loss(self, X, y, weights, bias):
        """
        Compute mean squared error loss.
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target values
        weights : np.array
            Model weights
        bias : float
            Model bias
        
        Returns:
        --------
        float : Loss value
        """
        n_samples = len(y)
        predictions = X.dot(weights) + bias
        loss = (1 / (2 * n_samples)) * np.sum((predictions - y) ** 2)
        return loss
    
    def _compute_gradients(self, X, y, weights, bias):
        """
        Compute gradients for weights and bias.
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target values
        weights : np.array
            Model weights
        bias : float
            Model bias
        
        Returns:
        --------
        tuple : Gradients for weights and bias
        """
        n_samples = len(y)
        predictions = X.dot(weights) + bias
        
        dw = (1 / n_samples) * X.T.dot(predictions - y)
        db = (1 / n_samples) * np.sum(predictions - y)
        
        return dw, db
    
    def batch_gradient_descent(self, X, y):
        """
        Batch Gradient Descent: Uses entire dataset for each iteration.
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target values
        
        Returns:
        --------
        self : GradientDescentOptimizer
            Returns self with fitted weights
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for iteration in range(self.max_iterations):
            # Compute gradients on entire dataset
            dw, db = self._compute_gradients(X, y, self.weights, self.bias)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Compute and store loss
            loss = self._compute_loss(X, y, self.weights, self.bias)
            self.loss_history.append(loss)
            
            # Check convergence
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return self
    
    def stochastic_gradient_descent(self, X, y):
        """
        Stochastic Gradient Descent: Uses one sample at a time.
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target values
        
        Returns:
        --------
        self : GradientDescentOptimizer
            Returns self with fitted weights
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for iteration in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Update using one sample at a time
            for i in range(n_samples):
                Xi = X_shuffled[i:i+1]
                yi = y_shuffled[i:i+1]
                
                dw, db = self._compute_gradients(Xi, yi, self.weights, self.bias)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Compute loss on full dataset
            loss = self._compute_loss(X, y, self.weights, self.bias)
            self.loss_history.append(loss)
            
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return self
    
    def mini_batch_gradient_descent(self, X, y, batch_size=32):
        """
        Mini-batch Gradient Descent: Uses small batches of data.
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target values
        batch_size : int, default=32
            Size of mini-batches
        
        Returns:
        --------
        self : GradientDescentOptimizer
            Returns self with fitted weights
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        for iteration in range(self.max_iterations):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Process mini-batches
            for i in range(0, n_samples, batch_size):
                Xi = X_shuffled[i:i+batch_size]
                yi = y_shuffled[i:i+batch_size]
                
                dw, db = self._compute_gradients(Xi, yi, self.weights, self.bias)
                
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            # Compute loss on full dataset
            loss = self._compute_loss(X, y, self.weights, self.bias)
            self.loss_history.append(loss)
            
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return self
    
    def momentum_gradient_descent(self, X, y, momentum=0.9):
        """
        Gradient Descent with Momentum.
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target values
        momentum : float, default=0.9
            Momentum coefficient
        
        Returns:
        --------
        self : GradientDescentOptimizer
            Returns self with fitted weights
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        # Initialize velocity
        velocity_w = np.zeros(n_features)
        velocity_b = 0
        
        for iteration in range(self.max_iterations):
            dw, db = self._compute_gradients(X, y, self.weights, self.bias)
            
            # Update velocity
            velocity_w = momentum * velocity_w - self.learning_rate * dw
            velocity_b = momentum * velocity_b - self.learning_rate * db
            
            # Update parameters
            self.weights += velocity_w
            self.bias += velocity_b
            
            loss = self._compute_loss(X, y, self.weights, self.bias)
            self.loss_history.append(loss)
            
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return self
    
    def adam_optimizer(self, X, y, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Adam (Adaptive Moment Estimation) optimizer.
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        y : np.array
            Target values
        beta1 : float, default=0.9
            Exponential decay rate for first moment estimates
        beta2 : float, default=0.999
            Exponential decay rate for second moment estimates
        epsilon : float, default=1e-8
            Small constant for numerical stability
        
        Returns:
        --------
        self : GradientDescentOptimizer
            Returns self with fitted weights
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.loss_history = []
        
        # Initialize moments
        m_w, v_w = np.zeros(n_features), np.zeros(n_features)
        m_b, v_b = 0, 0
        
        for iteration in range(self.max_iterations):
            dw, db = self._compute_gradients(X, y, self.weights, self.bias)
            
            # Update biased first moment estimate
            m_w = beta1 * m_w + (1 - beta1) * dw
            m_b = beta1 * m_b + (1 - beta1) * db
            
            # Update biased second raw moment estimate
            v_w = beta2 * v_w + (1 - beta2) * (dw ** 2)
            v_b = beta2 * v_b + (1 - beta2) * (db ** 2)
            
            # Compute bias-corrected moment estimates
            m_w_hat = m_w / (1 - beta1 ** (iteration + 1))
            m_b_hat = m_b / (1 - beta1 ** (iteration + 1))
            v_w_hat = v_w / (1 - beta2 ** (iteration + 1))
            v_b_hat = v_b / (1 - beta2 ** (iteration + 1))
            
            # Update parameters
            self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + epsilon)
            self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + epsilon)
            
            loss = self._compute_loss(X, y, self.weights, self.bias)
            self.loss_history.append(loss)
            
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance:
                    print(f"Converged at iteration {iteration}")
                    break
        
        return self
    
    def predict(self, X):
        """
        Make predictions using fitted model.
        
        Parameters:
        -----------
        X : np.array
            Feature matrix
        
        Returns:
        --------
        np.array : Predictions
        """
        return X.dot(self.weights) + self.bias


class LearningRateScheduler:
    """
    Learning rate scheduling strategies.
    """
    
    @staticmethod
    def step_decay(initial_lr, epoch, drop_rate=0.5, epochs_drop=10):
        """
        Step decay schedule.
        
        Parameters:
        -----------
        initial_lr : float
            Initial learning rate
        epoch : int
            Current epoch number
        drop_rate : float, default=0.5
            Rate to drop learning rate
        epochs_drop : int, default=10
            Drop learning rate every epochs_drop epochs
        
        Returns:
        --------
        float : Scheduled learning rate
        """
        return initial_lr * (drop_rate ** (epoch // epochs_drop))
    
    @staticmethod
    def exponential_decay(initial_lr, epoch, decay_rate=0.95):
        """
        Exponential decay schedule.
        
        Parameters:
        -----------
        initial_lr : float
            Initial learning rate
        epoch : int
            Current epoch number
        decay_rate : float, default=0.95
            Exponential decay rate
        
        Returns:
        --------
        float : Scheduled learning rate
        """
        return initial_lr * (decay_rate ** epoch)
    
    @staticmethod
    def cosine_annealing(initial_lr, epoch, total_epochs):
        """
        Cosine annealing schedule.
        
        Parameters:
        -----------
        initial_lr : float
            Initial learning rate
        epoch : int
            Current epoch number
        total_epochs : int
            Total number of epochs
        
        Returns:
        --------
        float : Scheduled learning rate
        """
        return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / total_epochs))


# Demonstrations

def demonstrate_gradient_descent_variants():
    """
    Demonstrate different gradient descent variants.
    """
    print("=" * 80)
    print("GRADIENT DESCENT VARIANTS DEMONSTRATION")
    print("=" * 80)
    
    # Generate sample data
    np.random.seed(42)
    X, y = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\n1. Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    
    # Test different optimizers
    optimizers = {
        'Batch GD': lambda: GradientDescentOptimizer(learning_rate=0.01, max_iterations=100),
        'Stochastic GD': lambda: GradientDescentOptimizer(learning_rate=0.001, max_iterations=50),
        'Mini-batch GD': lambda: GradientDescentOptimizer(learning_rate=0.01, max_iterations=100),
        'Momentum GD': lambda: GradientDescentOptimizer(learning_rate=0.01, max_iterations=100),
        'Adam': lambda: GradientDescentOptimizer(learning_rate=0.01, max_iterations=100)
    }
    
    results = {}
    
    print("\n2. Training with different optimizers:\n")
    
    # Batch GD
    print("   a) Batch Gradient Descent:")
    opt = optimizers['Batch GD']()
    opt.batch_gradient_descent(X, y)
    results['Batch GD'] = opt.loss_history
    print(f"      Final loss: {opt.loss_history[-1]:.4f}")
    
    # Stochastic GD
    print("\n   b) Stochastic Gradient Descent:")
    opt = optimizers['Stochastic GD']()
    opt.stochastic_gradient_descent(X, y)
    results['Stochastic GD'] = opt.loss_history
    print(f"      Final loss: {opt.loss_history[-1]:.4f}")
    
    # Mini-batch GD
    print("\n   c) Mini-batch Gradient Descent:")
    opt = optimizers['Mini-batch GD']()
    opt.mini_batch_gradient_descent(X, y, batch_size=32)
    results['Mini-batch GD'] = opt.loss_history
    print(f"      Final loss: {opt.loss_history[-1]:.4f}")
    
    # Momentum GD
    print("\n   d) Momentum Gradient Descent:")
    opt = optimizers['Momentum GD']()
    opt.momentum_gradient_descent(X, y, momentum=0.9)
    results['Momentum GD'] = opt.loss_history
    print(f"      Final loss: {opt.loss_history[-1]:.4f}")
    
    # Adam
    print("\n   e) Adam Optimizer:")
    opt = optimizers['Adam']()
    opt.adam_optimizer(X, y)
    results['Adam'] = opt.loss_history
    print(f"      Final loss: {opt.loss_history[-1]:.4f}")
    
    print("\n3. Convergence Comparison:")
    for name, loss_history in results.items():
        print(f"   {name}: {len(loss_history)} iterations to convergence")
    
    print("\n" + "=" * 80)
    
    return results


def demonstrate_learning_rate_scheduling():
    """
    Demonstrate learning rate scheduling strategies.
    """
    print("\n" + "=" * 80)
    print("LEARNING RATE SCHEDULING DEMONSTRATION")
    print("=" * 80)
    
    initial_lr = 0.1
    total_epochs = 100
    
    print(f"\n1. Initial Learning Rate: {initial_lr}")
    print(f"2. Total Epochs: {total_epochs}\n")
    
    schedulers = {
        'Step Decay': [],
        'Exponential Decay': [],
        'Cosine Annealing': []
    }
    
    for epoch in range(total_epochs):
        schedulers['Step Decay'].append(
            LearningRateScheduler.step_decay(initial_lr, epoch)
        )
        schedulers['Exponential Decay'].append(
            LearningRateScheduler.exponential_decay(initial_lr, epoch)
        )
        schedulers['Cosine Annealing'].append(
            LearningRateScheduler.cosine_annealing(initial_lr, epoch, total_epochs)
        )
    
    print("3. Learning Rate at Key Epochs:\n")
    key_epochs = [0, 25, 50, 75, 99]
    
    for epoch in key_epochs:
        print(f"   Epoch {epoch}:")
        for name, schedule in schedulers.items():
            print(f"      {name}: {schedule[epoch]:.6f}")
        print()
    
    print("=" * 80)
    
    return schedulers


if __name__ == "__main__":
    # Run demonstrations
    results = demonstrate_gradient_descent_variants()
    schedules = demonstrate_learning_rate_scheduling()
    
    print("\n✅ Module 4 Complete: Gradient Descent")
    print("\nKey Takeaways:")
    print("1. Batch GD is stable but slow for large datasets")
    print("2. SGD is fast but noisy, good for online learning")
    print("3. Mini-batch GD balances speed and stability")
    print("4. Momentum helps accelerate convergence")
    print("5. Adam adapts learning rates per parameter")
    print("6. Learning rate scheduling improves convergence")
