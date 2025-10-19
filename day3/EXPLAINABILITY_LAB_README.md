# Model Explainability and Regulation: SHAP and LIME Lab

## Overview

This hands-on Jupyter notebook demonstrates key concepts in machine learning model interpretability and explainability using real-world data. The lab covers:

1. **Introduction to Model Interpretability** - Understanding the need for explainable AI
2. **SHAP (SHapley Additive exPlanations)** - Theory and practical implementation
3. **LIME (Local Interpretable Model-agnostic Explanations)** - Local explanations for individual predictions
4. **Regulatory Considerations for AI Models** - Compliance, fairness, and transparency

## Dataset

The notebook uses the **Adult Income Dataset** (Census Income dataset) from the UCI Machine Learning Repository. This dataset is ideal for demonstrating explainability in a regulatory context as it:

- Involves sensitive attributes (age, sex, race, education)
- Has regulatory implications for fairness and non-discrimination
- Contains interpretable features for clear explanations
- Represents real-world use cases (credit scoring, hiring decisions)

**Task**: Predict whether a person makes over $50K a year based on census data.

## Prerequisites

Before running the notebook, ensure you have the following installed:

```bash
# Core ML libraries
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

# Explainability libraries
shap>=0.41.0
lime>=0.2.0

# Jupyter
jupyter>=1.0.0
notebook>=6.4.0
```

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/tatwan/advanced_machine_learning.git
   cd advanced_machine_learning
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install just the packages needed for this notebook:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn shap lime jupyter notebook
   ```

## Running the Notebook

1. **Start Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

2. **Open the notebook**:
   Navigate to `explainability_regulation_lab.ipynb` in the Jupyter interface

3. **Run the cells**:
   - Execute cells sequentially from top to bottom
   - Use `Shift+Enter` to run each cell
   - Or use "Cell" → "Run All" to execute all cells

## Notebook Structure

### Part 1: Setup and Data Loading (Cells 1-5)
- Import libraries
- Load Adult Income dataset from UCI repository
- Explore dataset structure and statistics

### Part 2: Introduction to Model Interpretability (Cells 6-7)
- Why explainable AI matters
- Types of interpretability (global vs. local)
- Tradeoff between accuracy and interpretability

### Part 3: Data Preprocessing and Model Training (Cells 8-9)
- Handle missing values
- Encode categorical variables
- Train Random Forest classifier
- Evaluate model performance

### Part 4: SHAP Demonstrations (Cells 10-17)
- Create SHAP explainer
- Global feature importance (summary plots, bar plots)
- Local explanations (force plots)
- Detailed prediction breakdowns
- Feature dependence plots and interactions

### Part 5: LIME Demonstrations (Cells 18-24)
- Create LIME explainer
- Local explanations for individual predictions
- Visualize feature contributions
- Compare multiple explanations
- SHAP vs LIME side-by-side comparison

### Part 6: Regulatory Considerations (Cells 25-31)
- Overview of regulatory frameworks (GDPR, ECOA, etc.)
- Protected attributes and fairness
- Bias detection using SHAP
- Generate adverse action notices
- Best practices for compliance

### Part 7: Summary and Exercises (Cells 32-43)
- Key takeaways
- When to use SHAP vs LIME
- Regulatory compliance checklist
- Practice exercises

## Key Concepts Covered

### SHAP (SHapley Additive exPlanations)
- **Theoretical Foundation**: Based on Shapley values from game theory
- **Strengths**: Consistent, theoretically grounded, both global and local
- **Use Cases**: Regulatory compliance, thorough model analysis
- **Visualizations**: Summary plots, force plots, dependence plots

### LIME (Local Interpretable Model-agnostic Explanations)
- **Approach**: Local linear approximation of complex models
- **Strengths**: Model-agnostic, fast, intuitive
- **Use Cases**: Quick explanations, text/image data
- **Visualizations**: Feature contribution bar charts

### Regulatory Compliance
- **GDPR**: Right to explanation for automated decisions
- **ECOA**: Adverse action notices for credit decisions
- **Fair Lending**: Non-discrimination requirements
- **Best Practices**: Documentation, monitoring, human oversight

## Expected Outputs

When you run the notebook, you'll see:

1. **Data Exploration**: Dataset statistics and distributions
2. **Model Performance**: Confusion matrix, accuracy metrics
3. **SHAP Visualizations**:
   - Global feature importance rankings
   - Individual prediction explanations
   - Feature interaction plots
4. **LIME Visualizations**:
   - Local feature contributions
   - Comparison across predictions
5. **Fairness Analysis**: Demographic group comparisons
6. **Adverse Action Notices**: Sample regulatory-compliant explanations

## Troubleshooting

### Dataset Loading Issues
If you have network issues loading the dataset from UCI:
```python
# Alternative: Download and load locally
# 1. Download from: https://archive.ics.uci.edu/ml/datasets/adult
# 2. Save as 'adult.data' in the same directory
# 3. Modify the loading code:
df = pd.read_csv('adult.data', names=column_names, na_values=' ?', skipinitialspace=True)
```

### SHAP/LIME Installation Issues
If you have trouble installing SHAP or LIME:
```bash
# Try installing with specific versions
pip install shap==0.41.0 lime==0.2.0.1

# Or use conda
conda install -c conda-forge shap lime
```

### Memory Issues
If you run out of memory:
- Reduce the sample size: `X_test_sample = X_test.sample(n=500)`
- Use fewer trees in Random Forest: `n_estimators=50`
- Close other applications

### Visualization Issues
If plots don't display:
```python
# Add at the beginning of the notebook
%matplotlib inline
```

## Learning Outcomes

After completing this notebook, you will be able to:

✅ Understand the importance of model interpretability and explainability  
✅ Use SHAP for global and local model explanations  
✅ Use LIME for quick local explanations  
✅ Compare and contrast SHAP and LIME approaches  
✅ Detect potential bias in ML models  
✅ Generate regulatory-compliant explanations  
✅ Apply explainability tools to real-world datasets  
✅ Understand regulatory requirements for AI models  

## Further Resources

- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [Interpretable ML Book](https://christophm.github.io/interpretable-ml-book/)
- [Fairlearn Library](https://fairlearn.org/)
- [EU AI Act](https://artificialintelligenceact.eu/)
- [GDPR Right to Explanation](https://gdpr-info.eu/)

## Practice Exercises

The notebook includes several exercises to deepen your understanding:

1. Compare different models (Gradient Boosting vs Random Forest)
2. Create interaction features and analyze their impact
3. Implement fairness improvements
4. Customize adverse action notices
5. Apply to different datasets (healthcare, credit scoring)
6. Analyze SHAP interaction values
7. Test LIME explanation stability

## Contributing

If you find issues or have suggestions for improvements:
1. Open an issue in the GitHub repository
2. Submit a pull request with your changes
3. Share your insights and learnings

## License

This educational material is provided for learning purposes as part of the Advanced Machine Learning course.

## Contact

For questions or feedback, please open an issue in the repository.

---

**Note**: This notebook requires an internet connection to download the dataset from UCI. All visualizations and explanations are generated dynamically based on the model's predictions.
