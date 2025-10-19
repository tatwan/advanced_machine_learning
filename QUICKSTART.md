# Advanced Machine Learning Course - Quick Start Guide

## Course Overview

This is a comprehensive 3-day advanced machine learning course with hands-on Python implementations. The course covers data management, model optimization, evaluation, ensemble methods, bias mitigation, probabilistic approaches, marketing analytics, explainability, and production ML.

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/tatwan/advanced_machine_learning.git
cd advanced_machine_learning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Course Structure

```
advanced_machine_learning/
├── day1/          # Data Management & Model Optimization
├── day2/          # Evaluation, Ensembles, Bias & Probabilistic Methods
├── day3/          # Marketing Analytics, Explainability & Production ML
├── data/          # Sample datasets
├── exercises/     # Exercise notebooks
└── solutions/     # Solution notebooks
```

### 3. Running the Modules

Each Python module can be run independently:

```bash
# Day 1 - Data Management
python day1/01_data_management.py

# Day 2 - Evaluation Metrics
python day2/01_evaluation_metrics.py

# Day 3 - Marketing Analytics
python day3/01_marketing_analytics.py
```

### 4. Running Jupyter Notebooks

For hands-on labs, launch Jupyter:

```bash
# Start Jupyter Notebook
jupyter notebook

# Or Jupyter Lab
jupyter lab

# Then open:
# - advanced_feature_engineering.ipynb
# - marketing_mix_modeling_and_attribution.ipynb
```

### 4. Running Jupyter Notebooks

Interactive hands-on labs are available as Jupyter notebooks:

```bash
# Start Jupyter Lab or Notebook
jupyter lab
# or
jupyter notebook

# Then open the notebook:
# - ensemble_methods_lab.ipynb - Comprehensive ensemble methods hands-on lab
# - advanced_feature_engineering.ipynb - Advanced feature engineering techniques
```

## Day-by-Day Breakdown

### Day 1: Data Management and Model Optimization (8 hours)

**Morning (4 hours)**
- Module 1: Data Management (`01_data_management.py`)
  - Handling dirty data, missing values
  - Data validation and cleaning
  
- Module 2: Outlier Detection (`02_outlier_detection.py`)
  - Statistical methods (Z-score, IQR)
  - ML-based methods (Isolation Forest, LOF)
  
- Module 3: Feature Engineering (`03_feature_engineering.py`)
  - Polynomial features, interactions
  - Feature selection methods

**Afternoon (4 hours)**
- Module 4: Gradient Descent (`04_gradient_descent.py`)
  - Batch, mini-batch, stochastic GD
  - Momentum, Adam optimization
  
- Module 5: SVMs and Kernels (`05_svm_and_kernels.py`)
  - Linear and non-linear SVMs
  - Kernel trick explained
  
- Module 6: Hyperparameter Tuning (`06_hyperparameter_tuning.py`)
  - Grid search, random search
  - Bayesian optimization, Optuna

### Day 2: Evaluation, Ensembles, Bias, and Probabilistic Methods (8 hours)

**Morning (4 hours)**
- Module 1: Evaluation Metrics (`01_evaluation_metrics.py`)
  - Classification and regression metrics
  - Custom business metrics
  
- Module 2: Random Forests (`02_random_forests.py`)
  - Bagging and bootstrap aggregating
  - Feature importance analysis
  
- Module 3: Gradient Boosting (`03_gradient_boosting.py`)
  - XGBoost, LightGBM, CatBoost
  - Library comparison

- **Hands-on Lab**: Ensemble Methods (`ensemble_methods_lab.ipynb`)
  - Using multiple models together
  - Random Forests and Gradient Boosted Trees
  - Bootstrap aggregation (bagging)
  - Combining heterogeneous models (stacking)
  - Comprehensive evaluation of ensembles

**Afternoon (4 hours)**
- Module 4: Bias Mitigation (`04_bias_mitigation.py`)
  - Fairness metrics and definitions
  - Bias detection and mitigation strategies
  
- Module 5: Probabilistic Methods (`05_probabilistic_methods.py`)
  - Bayesian inference and optimization
  - Uncertainty quantification

### Day 3: Marketing Analytics, Explainability, and Production ML (8 hours)

**Morning (4 hours)**
- Module 1: Marketing Analytics (`01_marketing_analytics.py`)
  - Media Mix Modeling (MMM)
  - Multi-Touch Attribution (MTA)
  - Customer Lifetime Value (CLV)
  
- **Hands-On Lab**: MMM & MTA (`marketing_mix_modeling_and_attribution.ipynb`)
  - Interactive Jupyter notebook with comprehensive examples
  - Real-world inspired datasets (retail/e-commerce simulation)
  - Budget optimization exercises
  - Comparison of attribution models
  - Visualizations and practical insights
  
- Module 2: SHAP Explainability (`02_shap_explainability.py`)
  - Shapley values explained
  - Global and local interpretability

**Afternoon (4 hours)**
- Module 3: LIME Explainability (`03_lime_explainability.py`)
  - Local interpretable explanations
  - LIME vs SHAP comparison
  
- Module 4: Production ML (`04_production_ml.py`)
  - Drift detection (data and concept)
  - Model monitoring and retraining
  - Production best practices

## Learning Objectives

By the end of this course, you will be able to:

1. ✅ Clean and prepare complex, real-world datasets
2. ✅ Implement and optimize advanced ML algorithms
3. ✅ Select appropriate evaluation metrics for different problems
4. ✅ Build and tune ensemble models
5. ✅ Identify and mitigate algorithmic bias
6. ✅ Apply probabilistic methods to ML problems
7. ✅ Implement marketing analytics solutions
8. ✅ Explain and interpret complex ML models
9. ✅ Deploy and maintain ML models in production
10. ✅ Monitor and retrain models to handle drift

## Prerequisites

- Strong understanding of basic machine learning concepts
- Python programming experience (intermediate level)
- Familiarity with NumPy, Pandas, and Scikit-learn
- Basic understanding of statistics and linear algebra

## Key Features

- **Hands-on**: Every module includes working code examples
- **Comprehensive**: Covers theory, implementation, and practical applications
- **Self-contained**: Each module can be run independently
- **Production-ready**: Focuses on real-world applications
- **Best practices**: Industry-standard approaches throughout

## Module Execution Time

Each module takes approximately 30-60 minutes to run through all demonstrations:
- Quick overview: 10-15 minutes
- Full walkthrough: 30-45 minutes
- With exercises: 60-90 minutes

## Tips for Success

1. **Follow the order**: Modules build on previous knowledge
2. **Run the code**: Don't just read - execute and experiment
3. **Modify examples**: Try different parameters and datasets
4. **Read comments**: Code is well-documented with explanations
5. **Take notes**: Each module ends with key takeaways

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)
- [Fairlearn Documentation](https://fairlearn.org/)

## Troubleshooting

### Common Issues

1. **Module not found errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Memory errors**: Some modules use large datasets
   - Reduce sample sizes in examples
   - Close other applications

3. **Slow execution**: Tree-based models can be slow
   - Reduce n_estimators
   - Use fewer iterations

## Support

For questions or issues:
- Open an issue in the GitHub repository
- Check module docstrings for detailed documentation
- Review the comprehensive README.md

## License

This course material is provided for educational purposes.

---

**Ready to Start?** Begin with Day 1, Module 1:
```bash
python day1/01_data_management.py
```

Good luck with your advanced ML journey! 🚀
