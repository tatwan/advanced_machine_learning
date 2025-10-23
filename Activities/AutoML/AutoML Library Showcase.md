# AutoML Notes

This notebook demonstrates three popular open-source AutoML libraries as alternatives to PyCaret:

1. **H2O AutoML** - Industry-standard distributed machine learning platform
2. **AutoGluon** - AWS-backed framework with state-of-the-art performance
3. **FLAML** - Microsoft's lightweight and efficient AutoML solution

## Installation

All required libraries have been pre-installed. If you need to install them in a different environment, use:

```bash
pip install h2o autogluon flaml scikit-learn pandas numpy
```

## Running the Notebook

### Option 1: Jupyter Lab (Recommended)
```bash
jupyter lab automl_showcase.ipynb
```

### Option 2: Jupyter Notebook
```bash
jupyter notebook automl_showcase.ipynb
```

### Option 3: VS Code
Open the `.ipynb` file directly in VS Code with the Jupyter extension installed.

## What Each Library Demonstrates

### H2O AutoML
- Automatic model training and hyperparameter tuning
- Model stacking and ensembling
- Leaderboard of best models
- Built-in model explainability
- Scalable for large datasets

### AutoGluon
- Multi-layered model ensembling
- State-of-the-art performance with minimal code
- Support for multimodal data (tabular, text, images)
- Automated hyperparameter tuning
- Deep learning integration

### FLAML
- Cost-effective hyperparameter optimization
- Budget-aware optimization strategies
- Fast and resource-efficient
- Suitable for rapid prototyping
- Integration with scikit-learn ecosystem

## Dataset

The notebook uses the classic **Iris dataset** for demonstration purposes. This is a simple multi-class classification problem with 150 samples and 4 features, making it ideal for educational purposes.

## Time Constraints

Each AutoML library is given a **120-second time budget** to find the best models. This ensures the notebook runs quickly while still demonstrating the capabilities of each library.

## Expected Output

Each section will:
1. Train multiple models automatically
2. Display a leaderboard or summary of model performance
3. Show predictions on the test set
4. Report accuracy metrics

## Notes

- H2O requires initialization of a local cluster (handled automatically in the notebook)
- AutoGluon creates a directory `ag_models` to store trained models
- FLAML creates a log file `flaml.log` to track the optimization process
- Remember to shutdown the H2O cluster at the end (optional cleanup cell provided)

## Comparison Summary

| Library | Developer | Best For | Performance | Resource Usage |
|---------|-----------|----------|-------------|----------------|
| H2O AutoML | H2O.ai | Enterprise, scalability | High | Medium-High |
| AutoGluon | AWS | Best accuracy, multimodal | Very High | High |
| FLAML | Microsoft | Speed, efficiency | High | Low |

## Educational Use

This notebook is designed for students learning about AutoML. It demonstrates:
- How different AutoML libraries approach the same problem
- The trade-offs between performance, speed, and resource usage
- Low-code approaches to machine learning
- Automated model selection and hyperparameter tuning
- Model evaluation and comparison

## Further Resources

- **H2O**: https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html
- **AutoGluon**: https://auto.gluon.ai/stable/index.html
- **FLAML**: https://microsoft.github.io/FLAML/



