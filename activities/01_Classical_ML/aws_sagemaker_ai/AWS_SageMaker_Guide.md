# AWS SageMaker AI: Complete Step-by-Step Guide

This guide walks you through training and deploying a machine learning model using **AWS SageMaker AI/Studio**. We'll use XGBoost for customer churn prediction as our example.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Setting Up SageMaker Studio](#2-setting-up-sagemaker-studio)
3. [Creating a Notebook](#3-creating-a-notebook)
4. [Understanding the Workflow](#4-understanding-the-workflow)
5. [Running the Example](#5-running-the-example)
6. [Deploying Your Model](#6-deploying-your-model)
7. [Making Predictions](#7-making-predictions)
8. [Cleanup (Important!)](#8-cleanup-important)
9. [Cost Considerations](#9-cost-considerations)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Prerequisites

Before starting, ensure you have:

### AWS Account Setup
- [ ] An active AWS account with billing enabled
- [ ] IAM user with SageMaker permissions (or use the root account for testing)
- [ ] An S3 bucket for storing data and model artifacts

### Required IAM Permissions
Your IAM role needs these policies:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess` (or scoped to your bucket)

> **💡 Tip**: For learning purposes, you can use the AWS Free Tier which includes limited SageMaker Studio usage.

---

## 2. Where to Run This Notebook ⚠️

### Recommended: AWS SageMaker Studio

**This notebook is designed to run in AWS SageMaker Studio**, not locally.

#### Why SageMaker Studio?
- ✅ **No credentials needed** - Automatically uses IAM role
- ✅ **Pre-configured environment** - All libraries pre-installed
- ✅ **Seamless integration** - Direct access to S3, training jobs, endpoints
- ✅ **No setup required** - Ready to run immediately

> [!IMPORTANT]
> While you *can* run this locally, it requires:
> - AWS CLI configured with credentials
> - IAM user with programmatic access
> - Manual installation of all dependencies
> - Proper AWS credentials configuration
>
> **For learning purposes, use SageMaker Studio.**

---

## 3. Credentials and Authentication

### When Running in SageMaker Studio (Recommended)

**No credentials needed!** SageMaker Studio automatically:
- Uses the execution role attached to your domain
- Handles all AWS API authentication
- Manages S3 access permissions

```python
# This line automatically gets your execution role:
role = get_execution_role()
# No .env files, no access keys needed!
```

### When Running Locally (Advanced)

If you choose to run locally, you need:

1. **AWS CLI installed and configured**
   ```bash
   aws configure
   ```

2. **Provide your credentials** when prompted:
   - AWS Access Key ID
   - AWS Secret Access Key
   - Default region (e.g., `us-east-1`)

3. **Alternative: Use environment variables**
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_DEFAULT_REGION="us-east-1"
   ```

4. **Or use `.env` file** (requires `python-dotenv`):
   ```
   AWS_ACCESS_KEY_ID=your-access-key
   AWS_SECRET_ACCESS_KEY=your-secret-key
   AWS_DEFAULT_REGION=us-east-1
   ```

> [!WARNING]
> Never commit credentials to Git! Add `.env` to your `.gitignore`.

---

## 4. Setting Up SageMaker Studio

### Step 4.1: Navigate to SageMaker

1. Log into the [AWS Console](https://console.aws.amazon.com)
2. Search for "**SageMaker**" in the search bar
3. Click on **Amazon SageMaker**

### Step 4.2: Create a SageMaker Domain (First Time Only)

If you haven't used SageMaker Studio before:

1. Click **"Set up for single user"** (Quick setup)
2. AWS will create:
   - A SageMaker Domain
   - An execution role with necessary permissions
   - Default storage settings
3. Wait 5-10 minutes for setup to complete

### Step 4.3: Launch SageMaker Studio

1. In the SageMaker console, click **"Studio"** in the left sidebar
2. Click **"Open Studio"** next to your user profile
3. SageMaker Studio will open in a new browser tab

> **⚠️ Note**: SageMaker Studio may take 2-3 minutes to load on first launch.

---

## 5. Creating and Uploading the Notebook

### Step 5.1: Upload This Notebook to Studio

**Do NOT run this notebook locally - upload it to SageMaker Studio:**

1. Download `xgboost_customer_churn_sagemaker.ipynb` to your computer
2. In SageMaker Studio, click the **folder icon** (File Browser) in the left sidebar
3. Click the **upload icon** (↑) at the top of the file browser
4. Select the notebook file from your computer
5. Wait for upload to complete
6. Double-click the notebook to open it

### Step 5.2: Alternative - Create a New Notebook (Optional)

1. In SageMaker Studio, click the **"+"** icon (Launcher)
2. Under **Notebooks**, click **"Notebook"**
3. Select a kernel:
   - Choose **"Python 3 (Data Science)"** or **"Python 3 (SageMaker Distribution)"**
4. Click **"Select"**

### Step 5.3: Select the Kernel

1. Double-click the uploaded notebook to open it
2. Ensure the kernel is running (check top-right corner)

---

## 6. Understanding the Workflow

The SageMaker workflow follows these steps:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   1. Prepare    │───▶│   2. Upload     │───▶│   3. Train      │
│      Data       │    │     to S3       │    │     Model       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                      │
                                                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   6. Cleanup    │◀───│   5. Predict    │◀───│   4. Deploy     │
│    Resources    │    │                 │    │    Endpoint     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components

| Component | Description |
|-----------|-------------|
| **S3** | Storage for training data and model artifacts |
| **SageMaker Training Job** | Managed compute for model training |
| **SageMaker Endpoint** | Real-time inference endpoint |
| **IAM Role** | Permissions for SageMaker to access S3 |

---

## 7. Running the Example

### Step 7.1: Execute the Notebook

Run each cell in order by pressing `Shift + Enter`:

1. **Import Libraries** - Load SageMaker SDK and dependencies
2. **Setup Session** - Initialize SageMaker session and role
3. **Load Data** - Download and explore the churn dataset
4. **Preprocess** - Clean and prepare data for training
5. **Upload to S3** - Store training data in your S3 bucket
6. **Train** - Launch a SageMaker training job
7. **Evaluate** - Check model performance metrics

### Step 7.2: Monitor Training

While training runs:

1. Go to **SageMaker Console** → **Training** → **Training jobs**
2. Click on your job to see:
   - Status (InProgress, Completed, Failed)
   - Logs (CloudWatch)
   - Resource utilization

Training typically takes **5-10 minutes** with the default settings.

---

## 8. Deploying Your Model

After training completes, deploy the model:

```python
# Deploy to a real-time endpoint
predictor = xgb.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large'
)
```

### What Happens During Deployment

1. SageMaker creates an endpoint configuration
2. Provisions compute instances
3. Loads the trained model
4. Exposes an HTTPS endpoint for predictions

> **⚠️ Warning**: Endpoints incur charges while running. See [Cleanup](#8-cleanup-important).

---

## 9. Making Predictions

### Real-Time Predictions

```python
import numpy as np

# Sample customer data
test_data = np.array([[128, 1, 124.0, 100, 114.2, 100, 195.5, 100, 9.6, 5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Get prediction
result = predictor.predict(test_data)
print(f"Churn Probability: {result}")
```

### Batch Transform (For Large Datasets)

For processing large batches:

```python
transformer = xgb.transformer(
    instance_count=1,
    instance_type='ml.m5.large'
)
transformer.transform(data=s3_test_data, content_type='text/csv')
```

---

## 10. Cleanup (Important!)

**⚠️ CRITICAL: Delete resources to avoid ongoing charges!**

### Step 10.1: Delete Endpoint

```python
# In your notebook
predictor.delete_endpoint()
```

Or via the console:
1. Go to **SageMaker Console** → **Inference** → **Endpoints**
2. Select your endpoint
3. Click **Delete**

### Step 10.2: Delete Endpoint Configuration

1. Go to **Inference** → **Endpoint configurations**
2. Select and delete the configuration

### Step 10.3: Clean Up S3 Data (Optional)

```python
import boto3
s3 = boto3.resource('s3')

# Empty and delete your bucket (be careful!)
bucket = s3.Bucket('your-bucket-name')
bucket.objects.delete()
```

### Step 10.4: Stop Studio Apps

1. In SageMaker Console, go to **Studio**
2. Click on your domain
3. Delete or stop any running apps

---

## 11. Cost Considerations

### Typical Costs

| Resource | Instance Type | Cost (approx.) |
|----------|--------------|----------------|
| **Training** | ml.m5.xlarge | ~$0.23/hour |
| **Endpoint** | ml.m5.large | ~$0.12/hour |
| **Studio** | ml.t3.medium | ~$0.05/hour |
| **S3 Storage** | - | ~$0.023/GB/month |

### Cost-Saving Tips

1. **Use Spot Instances** for training (up to 90% savings)
   ```python
   xgb = Estimator(..., use_spot_instances=True, max_wait=3600)
   ```

2. **Delete endpoints** immediately after testing

3. **Use SageMaker Serverless Inference** for low-traffic endpoints

4. **Monitor with AWS Budgets** - Set alerts for spending limits

---

## 12. Troubleshooting

### Common Issues

#### "ResourceLimitExceeded" Error
- **Cause**: Account limits on SageMaker instances
- **Solution**: Request a limit increase via AWS Support

#### "AccessDenied" Error
- **Cause**: IAM role lacks permissions
- **Solution**: Add `AmazonSageMakerFullAccess` policy to your role

#### Training Job Fails
1. Check CloudWatch logs:
   - Go to **CloudWatch** → **Log groups** → `/aws/sagemaker/TrainingJobs`
2. Look for Python errors or data format issues

#### Endpoint Takes Too Long
- Deployment can take 5-10 minutes
- Check endpoint status in the console

### Getting Help

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK Docs](https://sagemaker.readthedocs.io/)
- [AWS Free Tier](https://aws.amazon.com/free/) - Check what's included

---

## Next Steps

After completing this example:

1. **Try Different Algorithms**: Linear Learner, DeepAR, BlazingText
2. **Build a Pipeline**: Use SageMaker Pipelines for MLOps
3. **Experiment Tracking**: Use SageMaker Experiments
4. **Hyperparameter Tuning**: Automatic model optimization
5. **Model Registry**: Version and manage models

---

**Happy Learning! 🚀**
