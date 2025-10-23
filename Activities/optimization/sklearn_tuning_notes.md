# Scikit-Learn Hyperparameter Tuning Notes

## Key Concepts

### GridSearchCV
- Exhaustively considers all parameter combinations
- Best for smaller parameter spaces
- Guarantees finding the best combination within the grid
- Computational cost grows exponentially with number of parameters

### RandomizedSearchCV
- Samples a given number of candidates from parameter space
- Budget can be chosen independent of number of parameters
- More efficient when adding parameters that don't influence performance
- Uses distributions (scipy.stats: expon, gamma, uniform, loguniform, randint)
- Any function with rvs() method can be used for sampling

## When to Use Each

**GridSearchCV:**
- Small parameter spaces
- Want exhaustive search
- Computational resources available
- Need guaranteed optimal within grid

**RandomizedSearchCV:**
- Large parameter spaces
- Limited computational budget
- Many parameters to tune
- Want to explore diverse combinations efficiently

## Best Practices
- Use cross-validation for robust evaluation
- Specify scoring metric appropriate for problem
- Can specify multiple metrics for scoring parameter
- Use distributions that match parameter characteristics (log-uniform for learning rates, etc.)

