# SHAP Research Notes

## Key Concepts

### What is SHAP?
- SHAP (SHapley Additive exPlanations) is a game theoretic approach to explain the output of any machine learning model
- Based on Shapley values from cooperative game theory
- Provides fair allocation of credit for a model's output among its input features

### Core Properties
1. **Additive Nature**: SHAP values always sum up to the difference between the model output and expected value
2. **Local Interpretability**: Can explain individual predictions
3. **Global Interpretability**: Can aggregate individual explanations to understand overall model behavior
4. **Model-agnostic**: Can be applied to any machine learning model

### Key Visualizations
1. **Partial Dependence Plots**: Show how changing a feature impacts model output
2. **Scatter Plots**: Show SHAP values across the dataset for specific features
3. **Waterfall Plots**: Show how each feature contributes to a single prediction
4. **Summary Plots**: Show feature importance across all predictions

### Example Dataset
- California Housing Dataset (available via shap.datasets.california())
- 20,640 blocks of houses across California in 1990
- Goal: predict natural log of median home price
- 8 features: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

### Basic Usage Pattern
```python
import shap
import sklearn

# Load data
X, y = shap.datasets.california(n_points=1000)

# Train model
model = sklearn.linear_model.LinearRegression()
model.fit(X, y)

# Create explainer
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X)

# Visualize
shap.plots.scatter(shap_values[:, "FeatureName"])
```

## Installation
- Main package: `shap`
- Dependencies: scikit-learn, pandas, matplotlib, numpy

