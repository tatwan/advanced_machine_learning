# LIME Research Notes

## Key Concepts

### What is LIME?
- LIME (Local Interpretable Model-agnostic Explanations)
- Explains what machine learning classifiers (or models) are doing
- Can explain individual predictions for text classifiers, tabular data, or images
- Model-agnostic: works with any black box classifier
- Based on the paper "Why Should I Trust You?: Explaining the Predictions of Any Classifier"

### Core Approach
- **Local Linear Approximation**: While the model may be very complex globally, LIME approximates it around the vicinity of a particular instance
- **Perturbation-based**: Samples instances around the instance being explained
- **Weighted Learning**: Weights samples according to their proximity to the instance being explained
- **Sparse Linear Model**: Learns a sparse linear model as an explanation

### Key Features
1. **Model-agnostic**: Can explain any black box classifier
2. **Local Interpretability**: Focuses on explaining individual predictions
3. **Multiple Data Types**: Supports text, tabular data, and images
4. **Two or more classes**: Supports binary and multiclass classification
5. **Built-in scikit-learn support**: Easy integration with scikit-learn classifiers

### Installation
```bash
pip install lime
```

### Supported Use Cases
1. **Text Classification**: Explains which words contribute to predictions
2. **Tabular Data**: Explains feature contributions for numerical/categorical data
3. **Image Classification**: Explains which regions of an image contribute to predictions
4. **Regression**: Can also be used for regression tasks

### Basic Usage Pattern
- Requires classifier to implement a function that takes raw input and outputs probability for each class
- Generates HTML visualizations (can be embedded in Jupyter notebooks)
- Also supports matplotlib visualizations

### Key Visualizations
- **Text**: Shows positive/negative words with weights
- **Tabular**: Shows feature contributions with weights
- **Images**: Shows regions that support or contradict predictions

## Comparison with SHAP
- LIME: Local approximation, perturbation-based, faster for individual predictions
- SHAP: Game theory-based, guarantees consistency, better for global understanding
- Both are model-agnostic
- LIME focuses more on local fidelity, SHAP on theoretical properties




## LIME Tabular Implementation Details

### Key Components
1. **LimeTabularExplainer**: Main class for explaining tabular data
   - Requires training data to compute statistics
   - Computes mean, std, and discretizes features into quartiles
   - Can handle both continuous and categorical features

### Basic Usage for Tabular Data
```python
import lime
import lime.lime_tabular
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load data
iris = load_iris()
train, test, labels_train, labels_test = train_test_split(iris.data, iris.target, train_size=0.80)

# Train model
rf = RandomForestClassifier(n_estimators=500)
rf.fit(train, labels_train)

# Create explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    train, 
    feature_names=iris.feature_names, 
    class_names=iris.target_names, 
    discretize_continuous=True
)

# Explain instance
exp = explainer.explain_instance(test[i], rf.predict_proba, num_features=2, top_labels=1)

# Show explanation
exp.show_in_notebook(show_table=True, show_all=False)
```

### Key Parameters
- **discretize_continuous**: Whether to discretize continuous features (default: True)
- **num_features**: Number of features to include in explanation
- **top_labels**: Number of top predicted classes to explain
- **show_table**: Display feature values in table format
- **show_all**: Show all features or only those used in explanation

### Why Training Data is Needed
1. To scale the data for meaningful distance computation
2. To sample perturbed instances by sampling from Normal(0,1) and multiplying by std and adding back the mean

