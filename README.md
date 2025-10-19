# Advanced Machine Learning Course

A comprehensive 3-day advanced machine learning course covering data management, model optimization, evaluation metrics, ensemble methods, bias mitigation, probabilistic approaches, marketing analytics, explainability tools, and production ML practices.

## Course Overview

This course builds on foundational ML knowledge with hands-on Python applications across 3 days. Each day focuses on key advanced topics with practical implementations and real-world applications.

## Prerequisites

- Strong understanding of basic machine learning concepts
- Python programming experience
- Familiarity with NumPy, Pandas, and Scikit-learn
- Basic understanding of statistics and linear algebra

## Setup Instructions

1. Clone this repository:
```bash
git clone https://github.com/tatwan/advanced_machine_learning.git
cd advanced_machine_learning
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Course Structure

### 📓 Hands-On Lab Notebooks

- **[Linear Models: Optimization and Evaluation](linear_models_optimization_evaluation.ipynb)** - Comprehensive hands-on lab covering:
  - Gradient descent optimization techniques
  - L1, L2, and ElasticNet regularization
  - Support Vector Machines with different kernels
  - k-Folds cross-validation
  - Model evaluation metrics
  - Hyperparameter tuning with Grid Search
  - Threshold adjustment for business needs
  - Handling class imbalance (SMOTE, class weights)
  - Advanced metrics (ROC-AUC, PR curves)
  
  👉 See [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md) for detailed usage instructions.

- **[Advanced Feature Engineering](advanced_feature_engineering.ipynb)** - Data preprocessing and feature engineering techniques

---

### Day 1: Data Management and Model Optimization
**Duration**: 8 hours

#### Morning Session (4 hours)
1. **Data Management** (2 hours)
   - Handling dirty data and missing values
   - Outlier detection and treatment methods
   - Feature engineering techniques
   - Data preprocessing pipelines

2. **Model Optimization I** (2 hours)
   - Gradient descent variants (SGD, Mini-batch, Adam)
   - Learning rate scheduling
   - Hands-on: Implementing custom gradient descent

#### Afternoon Session (4 hours)
3. **Model Optimization II** (2 hours)
   - Support Vector Machines (SVMs)
   - Kernel methods and the kernel trick
   - Hyperparameter tuning strategies
   - Grid search vs. Random search vs. Bayesian optimization

4. **Hands-on Lab** (2 hours)
   - Complete data preprocessing pipeline
   - Model optimization exercises
   - **Recommended**: Work through `linear_models_optimization_evaluation.ipynb`

**Materials**: `day1/`

---

### Day 2: Evaluation, Ensembles, Bias, and Probabilistic Methods
**Duration**: 8 hours

#### Morning Session (4 hours)
1. **Evaluation Metrics** (1.5 hours)
   - Classification metrics (Precision, Recall, F1, ROC-AUC)
   - Regression metrics (RMSE, MAE, R²)
   - Custom metrics and business-specific evaluation

2. **Ensemble Methods** (2.5 hours)
   - Random Forests: Theory and practice
   - Gradient Boosted Trees (GBMs, XGBoost, LightGBM)
   - Stacking and blending techniques
   - **Hands-on Lab**: `ensemble_methods_lab.ipynb`
     - Using multiple models together
     - Bootstrap aggregation demonstration
     - Combining heterogeneous models
     - Comprehensive ensemble evaluation

#### Afternoon Session (4 hours)
3. **Bias Mitigation** (2 hours)
   - Understanding algorithmic bias
   - Fairness metrics and definitions
   - Bias detection and mitigation strategies
   - Case studies in fair ML

4. **Probabilistic Approaches** (2 hours)
   - Bayesian methods in ML
   - Probabilistic graphical models
   - Uncertainty quantification
   - Hands-on: Bayesian optimization

**Materials**: `day2/`

---

### Day 3: Marketing Analytics, Explainability, and Production ML
**Duration**: 8 hours

#### Morning Session (4 hours)
1. **Marketing Analytics** (2 hours)
   - Media Mix Modeling (MMM)
   - Multi-Touch Attribution (MTA)
   - Customer lifetime value prediction
   - Campaign optimization

2. **Explainability Tools** (2 hours)
   - Model interpretability vs. explainability
   - SHAP (SHapley Additive exPlanations)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Feature importance analysis

#### Afternoon Session (4 hours)
3. **Production ML** (3 hours)
   - Model deployment strategies
   - Drift detection (concept drift, data drift)
   - Model monitoring and logging
   - Automated retraining pipelines
   - A/B testing in production

4. **Final Lab & Wrap-up** (1 hour)
   - End-to-end ML project
   - Best practices review
   - Q&A session

**Materials**: `day3/`

---

## Repository Structure

```
advanced_machine_learning/
├── README.md                                      # This file
├── NOTEBOOK_GUIDE.md                              # Guide for the optimization notebook
├── requirements.txt                               # Python dependencies
├── linear_models_optimization_evaluation.ipynb    # Hands-on lab notebook
├── advanced_feature_engineering.ipynb             # Feature engineering notebook
├── day1/                                          # Day 1 materials
│   ├── 01_data_management.py
│   ├── 02_outlier_detection.py
│   ├── 03_feature_engineering.py
│   ├── 04_gradient_descent.py
│   ├── 05_svm_and_kernels.py
│   └── 06_hyperparameter_tuning.py
├── day2/                                          # Day 2 materials
│   ├── 01_evaluation_metrics.py
│   ├── 02_random_forests.py
│   ├── 03_gradient_boosting.py
│   ├── 04_bias_mitigation.py
│   └── 05_probabilistic_methods.py
├── day3/                                          # Day 3 materials
│   ├── 01_marketing_analytics.py
│   ├── 02_shap_explainability.py
│   ├── 03_lime_explainability.py
│   └── 04_production_ml.py
├── data/                                          # Sample datasets
├── exercises/                                     # Exercise notebooks
└── solutions/                                     # Solution notebooks
```

## Learning Outcomes

By the end of this course, participants will be able to:

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

## Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [SHAP Documentation](https://shap.readthedocs.io/)
- [LIME Documentation](https://lime-ml.readthedocs.io/)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This course material is provided for educational purposes.

## Contact

For questions or feedback, please open an issue in this repository.