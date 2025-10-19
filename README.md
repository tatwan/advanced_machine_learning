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

## Hands-On Lab Notebooks

This repository includes interactive Jupyter notebooks for hands-on practice, organized by day:

### Day 1 Notebooks

#### 📓 Linear Models: Optimization and Evaluation
**Notebook**: `day1/linear_models_optimization_evaluation.ipynb`

Comprehensive hands-on lab covering:
- Gradient descent optimization techniques
- L1, L2, and ElasticNet regularization
- Support Vector Machines with different kernels
- k-Folds cross-validation
- Model evaluation metrics
- Hyperparameter tuning with Grid Search
- Threshold adjustment for business needs
- Handling class imbalance (SMOTE, class weights)
- Advanced metrics (ROC-AUC, PR curves)

#### 📓 Advanced Feature Engineering
**Notebook**: `day1/advanced_feature_engineering.ipynb`

Topics covered:
- Handling dirty data and missing values
- Feature imputation techniques
- Creating engineered features
- Denoising with ML models

### Day 2 Notebooks

#### 📓 Ensemble Methods Lab
**Notebook**: `day2/ensemble_methods_lab.ipynb`

Hands-on lab featuring:
- Using multiple models together
- Random Forests and Gradient Boosted Trees
- Bootstrap aggregation (bagging)
- Combining heterogeneous models (stacking)
- Comprehensive ensemble evaluation

#### 📓 Marketing Mix Modeling and Attribution
**Notebook**: `day2/marketing_mix_modeling_and_attribution.ipynb`

Interactive lab covering:
- Media Mix Modeling (MMM)
- Multi-Touch Attribution (MTA)
- Real-world inspired datasets (retail/e-commerce simulation)
- Budget optimization exercises
- Comparison of attribution models

See `day2/MMM_MTA_README.md` for detailed instructions.

#### 📓 Probabilistic Methods Lab
**Notebook**: `day2/probabilistic_methods_lab.ipynb`

Topics covered:
- Bayesian methods in ML
- Probabilistic graphical models
- Uncertainty quantification

### Day 3 Notebooks

#### 📓 Explainability and Regulation Lab
**Notebook**: `day3/explainability_regulation_lab.ipynb`

Comprehensive lab covering:
- Model interpretability fundamentals
- SHAP (SHapley Additive exPlanations) with real-world dataset
- LIME (Local Interpretable Model-agnostic Explanations)
- Regulatory considerations (GDPR, ECOA, Fair Lending)
- Bias detection and fairness analysis
- Generating adverse action notices

**Dataset**: Adult Income dataset from UCI ML Repository

See `day3/EXPLAINABILITY_LAB_README.md` for detailed instructions.

#### 📓 Model Drift and Retraining
**Notebook**: `day3/model_drift_and_retraining.ipynb`

A comprehensive hands-on lab demonstrating:
- **Understanding Model Drift**: Concept drift vs. data drift with visual examples
- **Detecting Data Drift**: Statistical tests (KS-test), Population Stability Index (PSI)
- **Detecting Concept Drift**: Performance monitoring and degradation detection
- **Retraining Strategies**: Trigger-based, periodic, and incremental learning approaches
- **Automated Retraining Pipeline**: Complete implementation with monitoring and alerts

**Dataset**: Simulated credit card fraud detection (realistic patterns with temporal drift)

## Course Structure

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
   - **Recommended**: Work through `day1/linear_models_optimization_evaluation.ipynb`

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
   - **Hands-on Lab**: `day2/ensemble_methods_lab.ipynb`
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
   - **Hands-on Lab**: `day2/probabilistic_methods_lab.ipynb`

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
   - **Hands-on Lab**: `day2/marketing_mix_modeling_and_attribution.ipynb`

2. **Explainability Tools** (2 hours)
   - Model interpretability vs. explainability
   - SHAP (SHapley Additive exPlanations)
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Feature importance analysis
   - **Hands-on Lab**: `day3/explainability_regulation_lab.ipynb`

#### Afternoon Session (4 hours)
3. **Production ML** (3 hours)
   - Model deployment strategies
   - Drift detection (concept drift, data drift)
   - Model monitoring and logging
   - Automated retraining pipelines
   - A/B testing in production
   - **Hands-on Lab**: `day3/model_drift_and_retraining.ipynb`

4. **Final Lab & Wrap-up** (1 hour)
   - End-to-end ML project
   - Best practices review
   - Q&A session

**Materials**: `day3/`

---

## Repository Structure

```
advanced_machine_learning/
├── README.md                                    # This file
├── QUICKSTART.md                                # Quick start guide
├── requirements.txt                             # Python dependencies
├── day1/                                        # Day 1 materials
│   ├── 01_data_management.py
│   ├── 02_outlier_detection.py
│   ├── 03_feature_engineering.py
│   ├── 04_gradient_descent.py
│   ├── 05_svm_and_kernels.py
│   ├── 06_hyperparameter_tuning.py
│   ├── advanced_feature_engineering.ipynb      # Hands-on lab: Feature engineering
│   ├── linear_models_optimization_evaluation.ipynb  # Hands-on lab: Linear models
│   ├── Linear_Models_Reference.md
│   └── Linear_Models.MD
├── day2/                                        # Day 2 materials
│   ├── 01_evaluation_metrics.py
│   ├── 02_random_forests.py
│   ├── 03_gradient_boosting.py
│   ├── 04_bias_mitigation.py
│   ├── 05_probabilistic_methods.py
│   ├── ensemble_methods_lab.ipynb              # Hands-on lab: Ensemble methods
│   ├── marketing_mix_modeling_and_attribution.ipynb  # Hands-on lab: MMM & MTA
│   ├── MMM_MTA_README.md
│   └── probabilistic_methods_lab.ipynb         # Hands-on lab: Probabilistic methods
└── day3/                                        # Day 3 materials
    ├── 01_marketing_analytics.py
    ├── 02_shap_explainability.py
    ├── 03_lime_explainability.py
    ├── 04_production_ml.py
    ├── explainability_regulation_lab.ipynb     # Hands-on lab: Explainability
    ├── model_drift_and_retraining.ipynb        # Hands-on lab: Model drift & retraining
    ├── EXPLAINABILITY_LAB_README.md
    ├── IMPLEMENTATION_SUMMARY.md
    ├── INSTRUCTOR_GUIDE.md
    └── Model_Drift.md
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