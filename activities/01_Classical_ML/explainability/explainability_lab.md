# Post-Hoc Model Explainability Lab: PDP, PFI, and ALE

## Introduction to Model Explainability (XAI)

In the field of machine learning, models are often referred to as "black boxes" because their internal workings and decision-making processes can be opaque. **Explainable Artificial Intelligence (XAI)** is a set of tools and techniques that allows us to understand how a model arrives at a particular decision.

This lab focuses on **post-hoc explainability**, which involves analyzing a trained model *after* it has been built. We will explore three powerful, model-agnostic techniques:

1.  **Permutation Feature Importance (PFI):** A global method to understand which features are most important to the model's overall performance.
2.  **Partial Dependence Plots (PDP):** A global method to visualize the marginal effect of one or two features on the predicted outcome.
3.  **Accumulated Local Effects (ALE):** An improved global method over PDP that accounts for feature correlation.

## Setup and Data Preparation

We will use the California Housing dataset, a classic regression problem, and train a black-box Random Forest model.

### 1. Install and Import Libraries

We will use `scikit-learn` for the model and PFI/PDP, and `PyALE` for the ALE plots.

```python
# Install PyALE if you haven't already
# !pip install PyALE scikit-learn pandas matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from PyALE import ale
```

### 2. Load and Prepare Data

```python
# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the first few rows of the feature data
print("Features (X_train) head:")
print(X_train.head())
```

### 3. Train the Black-Box Model

We train a Random Forest Regressor, which is a powerful, non-linear model often considered a "black box."

```python
# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate the model (R-squared score)
r_sq = model.score(X_test, y_test)
print(f"\nModel R-squared on test set: {r_sq:.4f}")
```

---

## Section 1: Permutation Feature Importance (PFI)

### Theory

**Permutation Feature Importance (PFI)** measures the importance of a feature by calculating the increase in the model's prediction error after permuting (shuffling) the feature's values. A feature is considered "important" if shuffling its values significantly increases the model's error, as the model relied on that feature for accurate predictions.

### Implementation

We will use the `permutation_importance` function from `scikit-learn`.

```python
# Calculate Permutation Feature Importance
result = permutation_importance(
    model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error'
)

# Organize results into a DataFrame
sorted_idx = result.importances_mean.argsort()
pfi_df = pd.DataFrame({
    'Feature': X_test.columns[sorted_idx],
    'Importance Mean': result.importances_mean[sorted_idx],
    'Importance Std': result.importances_std[sorted_idx]
})

print("\nPermutation Feature Importance Results:")
print(pfi_df.sort_values(by='Importance Mean', ascending=False))

# Visualize the PFI results
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(pfi_df['Feature'], pfi_df['Importance Mean'], xerr=pfi_df['Importance Std'])
ax.set_xlabel("Mean decrease in score (Negative MSE)")
ax.set_title("Permutation Feature Importance")
plt.tight_layout()
plt.show()
```

### Exercise 1: Interpreting PFI

1.  Based on the plot, which feature is the **most** important for predicting house prices?
2.  What does a negative importance value (if any) imply about a feature? (Hint: It means the feature is not useful, and the random shuffling actually improved the score by chance).
3.  Try changing the `scoring` parameter to `'r2'` and re-run the PFI calculation. How do the results change?

---

## Section 2: Partial Dependence Plots (PDP)

### Theory

**Partial Dependence Plots (PDP)** show the average marginal effect of one or two features on the predicted outcome of a machine learning model. It works by fixing the feature(s) of interest to a grid of values and averaging the model's predictions over all other features.

**Limitation:** PDP assumes that the feature(s) of interest are **uncorrelated** with the other features. If features are highly correlated, the PDP will average predictions over unlikely or impossible data points, leading to potentially misleading results.

### Implementation: 1D PDP

We will plot the partial dependence for the two most important features identified by PFI.

```python
# Features to plot (e.g., the two most important)
features_to_plot_1d = ['MedInc', 'AveOccup']

# Create the PDP display
fig, ax = plt.subplots(figsize=(12, 5))
pdp_display = PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=features_to_plot_1d,
    feature_names=X_train.columns.tolist(),
    target=0,
    n_jobs=-1,
    grid_resolution=20,
    ax=ax
)
fig.suptitle("1D Partial Dependence Plots")
plt.tight_layout()
plt.show()
```

### Implementation: 2D PDP (Feature Interaction)

We can also plot the joint dependence of two features to visualize their interaction.

```python
# Features to plot for 2D interaction
features_to_plot_2d = [('MedInc', 'AveOccup')]

# Create the 2D PDP display
fig, ax = plt.subplots(figsize=(10, 8))
pdp_display_2d = PartialDependenceDisplay.from_estimator(
    model,
    X_train,
    features=features_to_plot_2d,
    feature_names=X_train.columns.tolist(),
    target=0,
    n_jobs=-1,
    grid_resolution=20,
    ax=ax
)
fig.suptitle("2D Partial Dependence Plot (Interaction)")
plt.tight_layout()
plt.show()
```

### Exercise 2: Interpreting PDP

1.  Describe the relationship between `MedInc` (Median Income) and the predicted house price based on the 1D PDP. Is it linear?
2.  Examine the 2D PDP for `MedInc` and `AveOccup` (Average Occupancy). Does the effect of `MedInc` on price appear to change significantly depending on the value of `AveOccup`? If so, describe the interaction.

---

## Section 3: Accumulated Local Effects (ALE)

### Theory

**Accumulated Local Effects (ALE)** plots are a more robust and less biased alternative to PDPs, especially when features are correlated. Instead of averaging predictions over the entire feature space, ALE calculates the *local* effect of a feature on the prediction and then accumulates these local effects. This means it only considers the effect of a feature *at the actual observed data points*, thus avoiding the unrealistic data points that can skew PDPs.

### Implementation

We will use the `PyALE` library to generate the ALE plots.

```python
# Generate 1D ALE plots for the same features as PDP
ale_medinc = ale(
    X=X_train,
    model=model,
    feature='MedInc',
    feature_type='numerical',
    grid_size=50,
    include_CI=True
)

ale_aveoccup = ale(
    X=X_train,
    model=model,
    feature='AveOccup',
    feature_type='numerical',
    grid_size=50,
    include_CI=True
)

# Plotting the ALE results (PyALE returns a dictionary with plot data)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot MedInc ALE
axes[0].plot(ale_medinc['x_values'], ale_medinc['ale'], label='ALE')
axes[0].fill_between(ale_medinc['x_values'], ale_medinc['lower_bound'], ale_medinc['upper_bound'], alpha=0.2, label='95% CI')
axes[0].set_title('ALE Plot for MedInc')
axes[0].set_xlabel('MedInc')
axes[0].set_ylabel('Accumulated Local Effect on Prediction')
axes[0].grid(True, linestyle='--', alpha=0.6)

# Plot AveOccup ALE
axes[1].plot(ale_aveoccup['x_values'], ale_aveoccup['ale'], label='ALE')
axes[1].fill_between(ale_aveoccup['x_values'], ale_aveoccup['lower_bound'], ale_aveoccup['upper_bound'], alpha=0.2, label='95% CI')
axes[1].set_title('ALE Plot for AveOccup')
axes[1].set_xlabel('AveOccup')
axes[1].set_ylabel('Accumulated Local Effect on Prediction')
axes[1].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()
```

### Exercise 3: Comparing PDP and ALE

1.  Compare the ALE plot for `MedInc` with its corresponding PDP. Are they similar? Why or why not?
2.  Compare the ALE plot for `AveOccup` with its corresponding PDP. Note any differences, especially in the shape or magnitude of the effect.
3.  Based on your observations, which technique (PDP or ALE) do you think provides a more reliable explanation for the effect of `AveOccup` on the predicted house price, and why?

---

## Conclusion and Further Reading

You have successfully implemented and interpreted three fundamental post-hoc explainability techniques: PFI, PDP, and ALE. Understanding these methods is crucial for building trust and gaining insights from complex machine learning models.

### Further Reading

*   **PiML Toolbox Documentation:** [https://selfexplainml.github.io/PiML-Toolbox/_build/html/guides/explain.html](https://selfexplainml.github.io/PiML-Toolbox/_build/html/guides/explain.html)
*   **Scikit-learn Model Inspection:** [https://scikit-learn.org/stable/inspection.html#](https://scikit-learn.org/stable/inspection.html#)
*   **Interpretable Machine Learning Book (Christoph Molnar):** A comprehensive resource for all XAI methods.
    *   Partial Dependence Plots: [https://christophm.github.io/interpretable-ml-book/pdp.html](https://christophm.github.io/interpretable-ml-book/pdp.html)
    *   Accumulated Local Effects: [https://christophm.github.io/interpretable-ml-book/ale.html](https://christophm.github.io/interpretable-ml-book/ale.html)
`
