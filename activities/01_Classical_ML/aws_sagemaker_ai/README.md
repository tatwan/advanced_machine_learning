# AWS SageMaker AI

This module demonstrates how to train and deploy machine learning models using **Amazon SageMaker AI**.

## Contents

| File | Description |
|------|-------------|
| [AWS_SageMaker_Guide.md](AWS_SageMaker_Guide.md) | Step-by-step guide for using SageMaker Studio |
| [xgboost_customer_churn_sagemaker.ipynb](xgboost_customer_churn_sagemaker.ipynb) | Complete notebook example |

## Example: Customer Churn Prediction

This example uses **XGBoost** to predict customer churn in a telecom dataset.

### What You'll Learn

- Setting up AWS SageMaker Studio
- **Uploading and running notebooks in SageMaker Studio** (not locally)
- Training models with built-in algorithms
- Deploying real-time inference endpoints
- Making predictions
- Cleaning up resources (to avoid charges)

## 🎯 Where to Run

**⚠️ IMPORTANT: This notebook must be uploaded to and run in AWS SageMaker Studio.**

### Why Not Run Locally?
- ❌ Requires AWS credentials configuration
- ❌ Manual library installation needed  
- ❌ More complex setup

### SageMaker Studio Benefits:
- ✅ **No credentials needed** - Uses IAM roles automatically
- ✅ **Pre-configured environment** - Everything ready to go
- ✅ **Seamless AWS integration** - Direct access to all services

> See [AWS_SageMaker_Guide.md](AWS_SageMaker_Guide.md) for detailed setup instructions.

## Quick Start

1. **Read the Guide**: Start with [AWS_SageMaker_Guide.md](AWS_SageMaker_Guide.md)
2. **Setup SageMaker Studio**: Create your domain (one-time setup)
3. **Upload the Notebook**: Upload `xgboost_customer_churn_sagemaker.ipynb` to Studio
4. **Run All Cells**: Execute each cell in order (Shift + Enter)
5. **Cleanup**: Delete endpoint and resources when done

## Prerequisites

- AWS Account with billing enabled
- IAM permissions for SageMaker and S3
- Basic Python/ML knowledge

## ⚠️ Cost Warning

Running this example will incur AWS charges. The guide includes:
- Cost estimates for each resource type
- Tips for reducing costs
- Cleanup instructions to stop charges

## Technologies Used

- AWS SageMaker
- Amazon S3
- XGBoost
- Python (pandas, numpy, scikit-learn)
