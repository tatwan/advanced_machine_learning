# Optuna Hyperparameter Optimization Notes

## Key Features

**Define-by-run API**: Optuna uses an imperative, define-by-run style that allows dynamic construction of search spaces. This means you can use Python conditionals and loops to define parameter spaces.

**Efficient Optimization Algorithms**: Adopts state-of-the-art algorithms for sampling hyperparameters and efficiently pruning unpromising trials. Uses Bayesian optimization approaches (TPE - Tree-structured Parzen Estimator by default).

**Easy Parallelization**: Can scale studies to tens or hundreds of workers with minimal code changes.

**Quick Visualization**: Provides plotting functions to inspect optimization histories.

**Lightweight**: Simple installation with few requirements, can handle a wide variety of tasks.

## Basic Concepts

- **Study**: An optimization based on an objective function
- **Trial**: A single execution of the objective function
- **Objective Function**: The function to minimize (or maximize) that evaluates model performance

## Key Advantages over Grid/Random Search

1. **Intelligent Sampling**: Uses Bayesian optimization (TPE) to intelligently sample hyperparameters based on previous trials
2. **Dynamic Search Spaces**: Can define conditional hyperparameters (e.g., different parameters for different model types)
3. **Pruning**: Can stop unpromising trials early to save computation
4. **Flexibility**: Easy to add/remove parameters without restructuring code

## When to Use Optuna

- Large hyperparameter spaces where grid search is infeasible
- When you want intelligent sampling based on previous results
- Complex search spaces with conditional parameters
- When computational budget is limited and you want to maximize efficiency
- When you need to prune unpromising trials early

