# Model Drift and Retraining Notebook Guide

## Overview

The `model_drift_and_retraining.ipynb` notebook provides a comprehensive, hands-on tutorial for understanding and implementing model drift detection and automated retraining pipelines in production machine learning systems.

## Learning Objectives

By the end of this notebook, you will be able to:

1. ✅ Distinguish between data drift and concept drift
2. ✅ Implement statistical tests for drift detection (KS-test, PSI)
3. ✅ Monitor model performance over time
4. ✅ Build automated retraining pipelines
5. ✅ Apply best practices for ML in production

## Prerequisites

- Python 3.7+
- Basic understanding of machine learning concepts
- Familiarity with scikit-learn
- Understanding of classification metrics

## Required Libraries

Install dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Key libraries used:
- `numpy` and `pandas` for data manipulation
- `scikit-learn` for ML models and metrics
- `matplotlib` and `seaborn` for visualization
- `scipy` for statistical tests

## Notebook Structure

### Section 1: Understanding Model Drift
- **Cells**: 1-4
- **Topics**: Introduction, types of drift (data vs. concept), visual demonstrations
- **Output**: Visualizations showing the difference between drift types

### Section 2: Dataset Preparation
- **Cells**: 5-7
- **Topics**: Simulated fraud detection dataset, exploratory data analysis
- **Output**: Dataset statistics, feature distributions

### Section 3: Baseline Model Training
- **Cells**: 8-9
- **Topics**: Training initial fraud detection model
- **Output**: Baseline performance metrics (F1, ROC-AUC, etc.)

### Section 4: Detecting Data Drift
- **Cells**: 10-13
- **Topics**: KS-test, PSI calculation, drift visualization
- **Output**: Drift detection results, PSI scores, distribution plots
- **Key Classes**: `DataDriftDetector`

### Section 5: Detecting Concept Drift
- **Cells**: 14-17
- **Topics**: Performance monitoring, degradation detection
- **Output**: Performance trends, drift alerts
- **Key Classes**: `ConceptDriftDetector`

### Section 6: Retraining Strategies
- **Cells**: 18-22
- **Topics**: Automated retraining pipeline, trigger-based retraining
- **Output**: Retraining decisions, performance improvements
- **Key Classes**: `AutomatedRetrainingPipeline`

### Section 7: Best Practices
- **Cell**: 23
- **Topics**: Production ML guidelines, monitoring, documentation

### Section 8: Summary and Exercises
- **Cell**: 24
- **Topics**: Key takeaways, practice exercises, additional resources

## Key Concepts Demonstrated

### Data Drift Detection Methods

1. **Kolmogorov-Smirnov Test**
   - Statistical test comparing distributions
   - Null hypothesis: samples from same distribution
   - Returns p-value for significance testing

2. **Population Stability Index (PSI)**
   - Industry standard for feature drift
   - PSI < 0.1: No significant change
   - 0.1 ≤ PSI < 0.25: Small change
   - PSI ≥ 0.25: Large change (retraining recommended)

### Concept Drift Detection

- Performance monitoring over time
- Rolling window accuracy analysis
- Threshold-based alerts
- Prediction distribution shifts

### Retraining Pipeline Components

1. **Monitoring**: Track data and performance metrics
2. **Detection**: Identify when drift occurs
3. **Decision**: Determine if retraining is needed
4. **Retraining**: Train new model on recent data
5. **Validation**: Ensure new model is better
6. **Deployment**: Update production model

## Dataset Description

The notebook uses a **simulated credit card fraud detection dataset** with the following characteristics:

- **Size**: 10,000+ transactions
- **Features**: 
  - `amount`: Transaction amount
  - `hour_of_day`: Time of transaction (0-24)
  - `distance_from_home`: Geographic distance
  - `transactions_last_24h`: Recent transaction count
  - `avg_transaction_amount`: Historical average
- **Target**: `is_fraud` (binary classification)
- **Class Balance**: ~2% fraud (realistic imbalance)

### Why This Dataset?

1. **Realistic Drift Simulation**: Easy to simulate temporal changes in fraud patterns
2. **Interpretable Features**: Clear relationship between features and fraud
3. **No Download Required**: Generated on-the-fly with controlled drift
4. **Educational Value**: Demonstrates real-world ML challenges

## Running the Notebook

### Option 1: Jupyter Notebook
```bash
jupyter notebook model_drift_and_retraining.ipynb
```

### Option 2: JupyterLab
```bash
jupyter lab
# Navigate to model_drift_and_retraining.ipynb
```

### Option 3: VS Code
1. Open the repository in VS Code
2. Install the Jupyter extension
3. Open `model_drift_and_retraining.ipynb`
4. Select Python kernel and run cells

## Expected Runtime

- **Total execution time**: ~5-10 minutes
- **Memory requirements**: ~500 MB
- **CPU**: Single core sufficient

Most cells execute in < 1 second. The longest operations are:
- Model training (~2-3 seconds per model)
- Visualization rendering (~1-2 seconds per plot)

## Common Issues and Solutions

### Issue 1: Missing Dependencies
**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```bash
pip install -r requirements.txt
```

### Issue 2: Matplotlib Display Issues
**Error**: Plots not showing in Jupyter

**Solution**: Add to first code cell:
```python
%matplotlib inline
```

### Issue 3: Memory Warnings
**Warning**: Large dataset warnings

**Solution**: The notebook uses reasonable data sizes. If issues persist:
- Reduce `n_samples` in dataset generation functions
- Use smaller `window_size` in monitoring

## Exercises and Extensions

After completing the notebook, try these exercises:

1. **Different Datasets**: Apply the pipeline to your own data
2. **Alternative Models**: Try different classifiers (XGBoost, LightGBM)
3. **Advanced Drift Detection**: Implement ADWIN or DDM algorithms
4. **Dashboard Creation**: Build a monitoring dashboard with Plotly
5. **Real-time Simulation**: Stream data to simulate production environment

## Additional Resources

### Related Course Materials
- `day3/04_production_ml.py`: Advanced production ML implementations
- `advanced_feature_engineering.ipynb`: Feature engineering techniques

### External Resources
- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [Alibi Detect](https://docs.seldon.io/projects/alibi-detect/)
- [MLOps Best Practices](https://ml-ops.org/)

### Academic Papers
- "Learning under Concept Drift: A Review" (2018)
- "A Survey on Concept Drift Adaptation" (2014)
- "Population Stability Index" (Financial Services Industry)

## Contributing

Found an issue or have a suggestion? Please:
1. Check existing issues in the repository
2. Open a new issue with detailed description
3. Include notebook cell number if applicable

## License

This educational material is part of the Advanced Machine Learning course.

---

**Last Updated**: 2025-10-19  
**Version**: 1.0  
**Author**: Advanced ML Course Team
