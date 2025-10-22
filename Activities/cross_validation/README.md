# Cross-Validation Notebooks

This folder contains three complementary notebooks that teach the fundamental concepts of cross-validation in machine learning. Each notebook builds upon the previous one, introducing increasingly advanced topics and real-world considerations.

##  Recommended Execution Order

### 1. **cross_validation_train_test.ipynb** (Foundations)
**Start here!** This notebook introduces the foundational concepts that underpin all cross-validation work.

**Key Learning Objectives:**
- Why we need to split data into training and testing sets
- The critical difference between training error and testing error
- Understanding overfitting and why it's problematic
- How cross-validation provides more robust model evaluation
- How to interpret cross-validation results and their variability

**Dataset:** California Housing (Regression problem)
- Predicting continuous house prices
- Real-world features with meaningful patterns
- Intuitive target variable (median house value)

**Topics Covered:**
- The central problem in machine learning: generalizing to unseen data
- Training error vs. testing error
- Why a simple train-test split is insufficient
- Introduction to K-Fold cross-validation
- Interpreting variability in cross-validation scores


---

### 2. **cross_validation_stratification.ipynb** (Classification Challenges)
 Now that you understand basic cross-validation, learn about special considerations for classification problems.

**Key Learning Objectives:**
- Why naive K-Fold splits can produce misleading results on ordered data
- The difference between shuffling and stratifying, and when to use each
- Using `StratifiedKFold` to preserve class proportions across folds
- Recognizing situations where stratification is not appropriate

**Dataset:** Iris (Classification problem)
- Classic multi-class classification task
- Demonstrates the dangers of ordered data
- Perfect toy dataset for understanding edge cases

**Topics Covered:**
- How K-Fold works without shuffling
- The problem of class imbalance across folds
- `StratifiedKFold` as a solution for classification
- Diagnosing unexpectedly low cross-validation scores
- When stratification is necessary vs. optional

**Important Insights:**
- The Iris dataset's ordered structure reveals pitfalls of naive splitting
- Class proportions matter in classification problems
- Stratification preserves class distribution across training and test folds


---

### 3. **cross_validation_grouping.ipynb** (Real-World Dependencies)
