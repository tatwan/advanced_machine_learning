# ML Metric Decision Tree

An interactive decision tree to help you choose the right evaluation metric for your machine learning problem.

## Live Demo

Check out the interactive version here 

https://tatwan.github.io/learn/model_evaluation_interactive.html

## How to Use

### Option 1: Online (GitHub Pages)
Click the link above to use the interactive tool directly in your browser.

### Option 2: Download and Use Locally
1. Download `index.html` from this repository
2. Open it in any modern web browser (Chrome, Firefox, Safari, Edge)
3. No internet connection required after download!

## What This Tool Covers

The decision tree guides you through choosing metrics for:

| Task Type | Example Metrics |
|-----------|----------------|
| **Binary Classification** | Accuracy, F1, Precision, Recall, AUC-ROC, AUC-PR, MCC |
| **Multi-Class Classification** | Balanced Accuracy, Macro/Micro F1, Cohen's Kappa |
| **Multi-Label Classification** | Hamming Loss, Jaccard Score, Subset Accuracy |
| **Regression** | RMSE, MAE, R-squared, MAPE |
| **Time Series Forecasting** | MASE, RMSSE, sMAPE, Directional Accuracy |
| **Ranking/Recommendations** | NDCG@K, MAP@K, MRR@K, Precision@K, Recall@K |
| **Clustering** | Silhouette, Davies-Bouldin, Calinski-Harabasz, ARI, AMI |
| **Object Detection** | mAP, IoU, AP@0.5, AP@0.75 |

## Key Features

- **Interactive Navigation**: Click through decision points to find the right metric
- **Detailed Explanations**: Each metric includes formulas, ranges, and use cases
- **Quick Reference Table**: Summary of all metrics at a glance
- **Imbalanced Data Warnings**: Highlights when common metrics (like accuracy) are misleading
- **Works Offline**: Single HTML file with no dependencies

## Setting Up GitHub Pages

To host this for your students:

1. Push this repository to GitHub
2. Go to **Settings** > **Pages**
3. Under "Source", select **Deploy from a branch**
4. Choose **main** branch and **/ (root)** folder
5. Click **Save**
6. Your site will be live at `https://YOUR_USERNAME.github.io/YOUR_REPO_NAME/`

## Files

- `index.html` - The interactive decision tree (standalone, works offline)
- `files/ML_Metric_Decision_Tree_v5_Complete.md` - Full text version of the decision tree
- `files/ml_metric_decision_tree.jsx` - React component source code

## Quick Reference

| Scenario | Recommended Metrics |
|----------|---------------------|
| Balanced binary classification | Accuracy, F1, AUC-ROC |
| Imbalanced binary classification | MCC, AUC-PR, F1 (avoid accuracy!) |
| False negatives are costly | Recall, F2-Score |
| False positives are costly | Precision, F0.5-Score |
| Multi-class with imbalance | Balanced Accuracy, Macro F1 |
| Regression with outliers | MAE, Median Absolute Error |
| Regression penalizing large errors | RMSE |
| Comparing across time series | MASE, RMSSE |
| Ranking top-K results | NDCG@K, MAP@K |
| Clustering without labels | Silhouette + Davies-Bouldin + Calinski-Harabasz |

## License

Free to use for educational purposes.

---

*Version 5 - Complete Edition*
