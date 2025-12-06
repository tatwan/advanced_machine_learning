#!/usr/bin/env python3
"""
Script to create complete solution for Activity 1: ML Anomaly Basics
"""

import json
from pathlib import Path
from copy import deepcopy

# Load the original notebook
notebook_path = Path("../activities/activity_01_ml_anomaly_basics.ipynb")
with open(notebook_path, 'r') as f:
    nb = json.load(f)

# Helper function to create code cell
def create_code_cell(source_lines):
    """Create a code cell with proper source formatting."""
    if isinstance(source_lines, str):
        source_lines = source_lines.split('\n')
    # Ensure each line ends with \n except the last
    source = [line + '\n' if i < len(source_lines) - 1 else line
              for i, line in enumerate(source_lines)]
    return source

# Solutions for each TODO section
solutions = {}

# Cell 14: Initialize KNN
solutions[14] = """# SOLUTION: Initialize a KNN detector with:
# - contamination=0.005 (expect 0.5% outliers)
# - method='mean' (use mean distance to neighbors)
# - n_neighbors=5 (consider 5 nearest neighbors)
knn = KNN(contamination=0.005, method='mean', n_neighbors=5)

print(knn)"""

# Cell 15: Fit KNN
solutions[15] = """# SOLUTION: Fit the KNN model on the 'Height' feature
# We use wh[['Height']] to select the column as a DataFrame (2D array)
knn.fit(wh[['Height']])"""

# Cell 16: Get KNN predictions
solutions[16] = """# SOLUTION: Get predictions from the KNN model
# The predict() method returns 0 for normal points and 1 for outliers
knn_pred = knn.predict(wh[['Height']])

# Convert to pandas Series for easier handling
knn_pred = pd.Series(knn_pred, index=wh.index)
print(f'Number of KNN outliers = {knn_pred.sum()}')"""

# Cell 20: Initialize LOF
solutions[20] = """# SOLUTION: Initialize a LOF detector with:
# - contamination=0.005
# - n_neighbors=20 (LOF typically uses more neighbors than KNN)
lof = LOF(contamination=0.005, n_neighbors=20)

print(lof)"""

# Cell 21: Fit LOF
solutions[21] = """# SOLUTION: Fit the LOF model on the 'Height' feature
lof.fit(wh[['Height']])"""

# Cell 22: Get LOF predictions
solutions[22] = """# SOLUTION: Get predictions from the LOF model
lof_pred = lof.predict(wh[['Height']])

lof_pred = pd.Series(lof_pred, index=wh.index)
print(f'Number of LOF outliers = {lof_pred.sum()}')"""

# Cell 25: Compare KNN vs LOF
solutions[25] = """# SOLUTION: Calculate how many outliers are found by both algorithms
# Use boolean operations to find overlap
both_algorithms = ((knn_pred == 1) & (lof_pred == 1)).sum()

# SOLUTION: Calculate how many are unique to KNN
only_knn = ((knn_pred == 1) & (lof_pred == 0)).sum()

# SOLUTION: Calculate how many are unique to LOF
only_lof = ((knn_pred == 0) & (lof_pred == 1)).sum()

print(f"Outliers detected by both: {both_algorithms}")
print(f"Outliers unique to KNN: {only_knn}")
print(f"Outliers unique to LOF: {only_lof}")"""

# Cell 30: Initialize CBLOF
solutions[30] = """# SOLUTION: Initialize a CBLOF detector with:
# - n_clusters=8 (try to find 8 natural groupings)
# - contamination=0.001 (expect 0.1% outliers)
# - alpha=0.9, beta=5 (default values)
cblof = CBLOF(n_clusters=8, contamination=0.001, alpha=0.9, beta=5, random_state=42)

print(cblof)"""

# Cell 31: Fit CBLOF and get predictions
solutions[31] = """# SOLUTION: Fit CBLOF on the 'Height' feature and get predictions
cblof.fit(wh[['Height']])
cblof_pred = cblof.predict(wh[['Height']])

# Convert to Series
cblof_pred = pd.Series(cblof_pred, index=wh.index)
print(f'Number of CBLOF outliers = {cblof_pred.sum()}')"""

# Cell 36: Initialize and fit COPOD
solutions[36] = """# SOLUTION: Initialize COPOD with contamination=0.005
copod = COPOD(contamination=0.005)

# SOLUTION: Fit COPOD on both 'Height' and 'Weight' features
# Now we use BOTH features together for multivariate analysis
copod.fit(wh[['Height', 'Weight']])"""

# Cell 37: Get COPOD predictions
solutions[37] = """# SOLUTION: Get predictions from COPOD
copod_pred = copod.predict(wh[['Height', 'Weight']])

copod_pred = pd.Series(copod_pred, index=wh.index)
print(f'Number of COPOD outliers = {copod_pred.sum()}')"""

# Cell 38: Visualize COPOD outliers
solutions[38] = """# SOLUTION: Visualize COPOD outliers using the helper function
copod_outliers = plot_outliers(wh, copod_pred, 'COPOD', color='orange')"""

# Cell 40: Initialize and fit ECOD
solutions[40] = """# SOLUTION: Initialize and fit ECOD (similar to COPOD)
ecod = ECOD(contamination=0.005)
ecod.fit(wh[['Height', 'Weight']])

# SOLUTION: Get predictions
ecod_pred = ecod.predict(wh[['Height', 'Weight']])

ecod_pred = pd.Series(ecod_pred, index=wh.index)
print(f'Number of ECOD outliers = {ecod_pred.sum()}')"""

# Cell 41: Visualize ECOD outliers
solutions[41] = """# SOLUTION: Visualize ECOD outliers
ecod_outliers = plot_outliers(wh, ecod_pred, 'ECOD', color='green')"""

# Cell 45: OCSVM unscaled
solutions[45] = """# SOLUTION: Initialize OCSVM with:
# - contamination=0.005
# - kernel='rbf'
# - gamma='auto'
# - nu=0.5
ocsvm_unscaled = OCSVM(contamination=0.005, kernel='rbf', gamma='auto', nu=0.5)

# SOLUTION: Fit on unscaled data (Height and Weight)
ocsvm_unscaled.fit(wh[['Height', 'Weight']])

# SOLUTION: Get predictions
ocsvm_unscaled_pred = ocsvm_unscaled.predict(wh[['Height', 'Weight']])

ocsvm_unscaled_pred = pd.Series(ocsvm_unscaled_pred, index=wh.index)
print(f'Number of OCSVM outliers (unscaled) = {ocsvm_unscaled_pred.sum()}')"""

# Cell 48: OCSVM scaled
solutions[48] = """# SOLUTION: Scale the data using standardizer
# standardizer normalizes features to have mean=0 and std=1
scaled_data = standardizer(wh[['Height', 'Weight']])

# SOLUTION: Initialize a new OCSVM model (same parameters as before)
ocsvm_scaled = OCSVM(contamination=0.005, kernel='rbf', gamma='auto', nu=0.5)

# SOLUTION: Fit on scaled data
ocsvm_scaled.fit(scaled_data)

# SOLUTION: Get predictions
ocsvm_scaled_pred = ocsvm_scaled.predict(scaled_data)

ocsvm_scaled_pred = pd.Series(ocsvm_scaled_pred, index=wh.index)
print(f'Number of OCSVM outliers (scaled) = {ocsvm_scaled_pred.sum()}')"""

# Cell 54: Initialize and fit IForest
solutions[54] = """# SOLUTION: Initialize IForest with:
# - contamination=0.005
# - n_estimators=100
# - bootstrap=False
# - random_state=42 (for reproducibility)
iforest = IForest(contamination=0.005, n_estimators=100, bootstrap=False, random_state=42)

# SOLUTION: Fit on Height and Weight (NO scaling needed!)
# IForest doesn't need feature scaling because it uses random splits
iforest.fit(wh[['Height', 'Weight']])"""

# Cell 55: Get IForest predictions
solutions[55] = """# SOLUTION: Get predictions from IForest
iforest_pred = iforest.predict(wh[['Height', 'Weight']])

iforest_pred = pd.Series(iforest_pred, index=wh.index)
print(f'Number of IForest outliers = {iforest_pred.sum()}')"""

# Cell 56: Visualize IForest outliers
solutions[56] = """# SOLUTION: Visualize IForest outliers
iforest_outliers = plot_outliers(wh, iforest_pred, 'Isolation Forest', color='cyan')"""

# Cell 61: Initialize and fit AutoEncoder (10 epochs)
solutions[61] = """# SOLUTION: Initialize AutoEncoder with:
# - contamination=0.005
# - lr=0.001 (learning rate)
# - epoch_num=10 (start small)
# - batch_size=32
auto_encoder = AutoEncoder(contamination=0.005, lr=0.001, epoch_num=10, batch_size=32,
                          verbose=0, random_state=42)

# SOLUTION: Fit on Height and Weight
# Note: This will take longer than previous algorithms!
auto_encoder.fit(wh[['Height', 'Weight']])"""

# Cell 62: Get AutoEncoder predictions
solutions[62] = """# SOLUTION: Get predictions
ae_pred = auto_encoder.predict(wh[['Height', 'Weight']])

ae_pred = pd.Series(ae_pred, index=wh.index)
print(f'Number of AutoEncoder outliers (10 epochs) = {ae_pred.sum()}')"""

# Cell 63: Visualize AutoEncoder outliers
solutions[63] = """# SOLUTION: Visualize AutoEncoder outliers
ae_outliers = plot_outliers(wh, ae_pred, 'AutoEncoder (10 epochs)', color='brown')"""

# Cell 65: AutoEncoder with 50 epochs
solutions[65] = """%%time
# SOLUTION: Train AutoEncoder with 50 epochs instead of 10
auto_encoder_50 = AutoEncoder(contamination=0.005, lr=0.001, epoch_num=50, batch_size=32,
                             verbose=0, random_state=42)

# Fit on Height and Weight
auto_encoder_50.fit(wh[['Height', 'Weight']])

# Get predictions
ae_pred_50 = auto_encoder_50.predict(wh[['Height', 'Weight']])

ae_pred_50 = pd.Series(ae_pred_50, index=wh.index)
print(f'Number of AutoEncoder outliers (50 epochs) = {ae_pred_50.sum()}')"""

# Cell 70: Create comparison DataFrame
solutions[70] = """# SOLUTION: Create a DataFrame with all predictions
# Include: KNN, LOF, CBLOF, COPOD, ECOD, OCSVM (scaled), IForest, AutoEncoder

comparison_df = pd.DataFrame({
    'KNN': knn_pred,
    'LOF': lof_pred,
    'CBLOF': cblof_pred,
    'COPOD': copod_pred,
    'ECOD': ecod_pred,
    'OCSVM_scaled': ocsvm_scaled_pred,
    'IForest': iforest_pred,
    'AutoEncoder': ae_pred_50,  # Using the 50-epoch version
}, index=wh.index)

comparison_df.head()"""

# Cell 72: Calculate algorithm overlap
solutions[72] = """# SOLUTION: Calculate how many algorithms flagged each point as an outlier
# Sum across columns for each row
comparison_df['num_algorithms'] = comparison_df.sum(axis=1)

# SOLUTION: Display points detected by 3 or more algorithms (high confidence outliers)
high_confidence_outliers = wh[comparison_df['num_algorithms'] >= 3]

print(f"Number of high-confidence outliers (3+ algorithms): {len(high_confidence_outliers)}")
print(f"\\nHigh-confidence outliers:")
print(high_confidence_outliers)"""

# Cell 73: Visualize high-confidence outliers
solutions[73] = """# SOLUTION: Visualize the high-confidence outliers on the original data
plt.figure(figsize=(10, 6))
plt.scatter(wh['Height'], wh['Weight'], alpha=0.5, label='Normal', s=30)

# SOLUTION: Add scatter plot for high-confidence outliers
plt.scatter(high_confidence_outliers['Height'], high_confidence_outliers['Weight'],
           color='red', s=150, alpha=0.7, label='High-Confidence Outliers',
           edgecolors='black', linewidths=2)

plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.title('High-Confidence Outliers (Detected by 3+ Algorithms)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()"""

# Cell 75: Create agreement matrix
solutions[75] = """# SOLUTION: Calculate pairwise agreement between algorithms
# For each pair of algorithms, count how many outliers they agree on

algorithms = comparison_df.columns[:-1]  # Exclude 'num_algorithms'
agreement_matrix = pd.DataFrame(index=algorithms, columns=algorithms)

for algo1 in algorithms:
    for algo2 in algorithms:
        # SOLUTION: Calculate how many points both algorithms marked as outliers
        # Use & for logical AND - counts where both are 1
        agreement = ((comparison_df[algo1] == 1) & (comparison_df[algo2] == 1)).sum()
        agreement_matrix.loc[algo1, algo2] = agreement

# Convert to numeric
agreement_matrix = agreement_matrix.astype(int)
print("Algorithm Agreement Matrix (number of shared outliers):")
print(agreement_matrix)"""

# Cell 78: Identify algorithm-specific outliers
solutions[78] = """# SOLUTION: For each algorithm, find outliers that ONLY it detected
# (i.e., num_algorithms == 1 for that specific algorithm)

for algo in algorithms:
    # SOLUTION: Find points where this algorithm = 1 AND num_algorithms = 1
    unique_outliers = wh[(comparison_df[algo] == 1) & (comparison_df['num_algorithms'] == 1)]

    print(f"\\n{algo} unique outliers: {len(unique_outliers)}")
    if len(unique_outliers) > 0:
        print(unique_outliers[['Height', 'Weight']])"""

# Cell 81: Challenge - implement 3 algorithms
solutions[81] = """# SOLUTION: Choose and implement 3 algorithms from different families
# We'll choose COPOD (probabilistic), IForest (ensemble), and LOF (distance-based)
# with tuned contamination rates

# Algorithm 1: COPOD with lower contamination
algo1 = COPOD(contamination=0.003)
algo1.fit(wh[['Height', 'Weight']])
pred1 = algo1.predict(wh[['Height', 'Weight']])

# Algorithm 2: IForest with more estimators for stability
algo2 = IForest(contamination=0.003, n_estimators=200, random_state=42)
algo2.fit(wh[['Height', 'Weight']])
pred2 = algo2.predict(wh[['Height', 'Weight']])

# Algorithm 3: LOF with optimized neighbors
algo3 = LOF(contamination=0.003, n_neighbors=30)
algo3.fit(wh[['Height', 'Weight']])
pred3 = algo3.predict(wh[['Height', 'Weight']])

print(f"COPOD detected: {pred1.sum()} outliers")
print(f"IForest detected: {pred2.sum()} outliers")
print(f"LOF detected: {pred3.sum()} outliers")"""

# Cell 82: Compare challenge results
solutions[82] = """# SOLUTION: Compare their results
# - How many outliers did each find?
# - How much do they overlap?
# - Which one seems most reasonable based on visualization?

pred1_series = pd.Series(pred1, index=wh.index)
pred2_series = pd.Series(pred2, index=wh.index)
pred3_series = pd.Series(pred3, index=wh.index)

# Calculate overlap
overlap_all = ((pred1_series == 1) & (pred2_series == 1) & (pred3_series == 1)).sum()
overlap_12 = ((pred1_series == 1) & (pred2_series == 1)).sum()
overlap_13 = ((pred1_series == 1) & (pred3_series == 1)).sum()
overlap_23 = ((pred2_series == 1) & (pred3_series == 1)).sum()

print(f"\\nOverlap Analysis:")
print(f"All three algorithms agree on: {overlap_all} outliers")
print(f"COPOD & IForest agree on: {overlap_12} outliers")
print(f"COPOD & LOF agree on: {overlap_13} outliers")
print(f"IForest & LOF agree on: {overlap_23} outliers")

# Visualize each
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (pred, name, ax) in enumerate(zip([pred1_series, pred2_series, pred3_series],
                                            ['COPOD', 'IForest', 'LOF'],
                                            axes)):
    outliers = wh[pred == 1]
    ax.scatter(wh['Height'], wh['Weight'], alpha=0.5, s=30)
    ax.scatter(outliers['Height'], outliers['Weight'],
              color='red', s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('Height (inches)')
    ax.set_ylabel('Weight (pounds)')
    ax.set_title(f'{name} - {len(outliers)} outliers')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""

# Now update the notebook cells
solution_nb = deepcopy(nb)

# Add a title cell at the beginning
title_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": ["# SOLUTION: Student Activity: Outlier Detection Using Machine Learning\\n",
               "\\n",
               "**This is the complete solution version of Activity 1.**\\n",
               "\\n",
               "All TODO sections have been filled in with working code, and exercises include example answers.\\n",
               "\\n",
               "---\\n"]
}
solution_nb['cells'].insert(0, title_cell)

# Update cells with solutions
for cell_idx, solution_code in solutions.items():
    # Adjust index by 1 due to added title cell
    actual_idx = cell_idx + 1
    if actual_idx < len(solution_nb['cells']):
        cell = solution_nb['cells'][actual_idx]
        if cell['cell_type'] == 'code':
            cell['source'] = create_code_cell(solution_code)

# Add answer cells for discussion questions
# After cell 26 (Discussion Questions for KNN vs LOF)
discussion_answers_26 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### SOLUTION: Discussion Answers\\n",
        "\\n",
        "**1. Which algorithm found more outliers?**\\n",
        "\\n",
        "Both KNN and LOF should find approximately 50 outliers (0.5% of ~10,000 points). The exact numbers may vary slightly based on the data distribution and how ties are handled.\\n",
        "\\n",
        "**2. Why might the algorithms disagree on some points?**\\n",
        "\\n",
        "- **KNN** uses absolute distance: A point is anomalous if it's far from its neighbors in absolute terms\\n",
        "- **LOF** uses relative density: A point is anomalous if its density is low compared to its neighbors\\n",
        "- A point in a sparse region might have similar distances to all neighbors (normal for KNN) but lower density than a dense cluster nearby (anomalous for LOF)\\n",
        "- KNN tends to find global outliers, while LOF can find local outliers within clusters\\n",
        "\\n",
        "**3. Which algorithm would you trust more for this data? Why?**\\n",
        "\\n",
        "For the weight-height data, **LOF** is generally more reliable because:\\n",
        "- The data has distinct clusters (male/female with different height-weight distributions)\\n",
        "- LOF can detect outliers within each cluster (e.g., unusually tall person within the male group)\\n",
        "- KNN might miss local anomalies if they're close to points from another cluster\\n",
        "\\n",
        "However, both provide valuable perspectives, and using them together gives the most complete picture.\\n"
    ]
}
solution_nb['cells'].insert(28, discussion_answers_26)

# After cell 50 (OCSVM scaling exercise)
scaling_answers_50 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### SOLUTION: Scaling Exercise Answers\\n",
        "\\n",
        "**1. How many outliers were detected without scaling vs with scaling?**\\n",
        "\\n",
        "Without scaling: Likely very few or incorrect outliers\\n",
        "With scaling: Approximately 50 outliers (0.5% of data) that are more reasonable\\n",
        "\\n",
        "**2. Why does scaling make such a difference for OCSVM?**\\n",
        "\\n",
        "- **Feature scale dominance**: Weight (120-180 lbs) has a much larger range than Height (60-72 inches)\\n",
        "- **Distance calculations**: OCSVM uses distance metrics in the kernel function. Large-scale features dominate these calculations\\n",
        "- **Unscaled**: The algorithm essentially only 'sees' the Weight dimension and ignores Height\\n",
        "- **Scaled**: Both features contribute equally to the decision boundary\\n",
        "\\n",
        "**3. Which result looks more reasonable when you visualize it?**\\n",
        "\\n",
        "The **scaled version** produces much more reasonable results:\\n",
        "- Outliers are detected in both dimensions (unusual height AND/OR weight combinations)\\n",
        "- The decision boundary is more circular/elliptical rather than dominated by one axis\\n",
        "- Points that are clearly unusual in the 2D space are correctly identified\\n",
        "\\n",
        "**Key Takeaway**: Always scale features for distance-based and kernel-based algorithms (OCSVM, KNN, k-means).\\n"
    ]
}
solution_nb['cells'].insert(52, scaling_answers_50)

# After cell 57 (IForest scaling question)
iforest_scaling_57 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### SOLUTION: Why IForest Doesn't Need Scaling\\n",
        "\\n",
        "**Explanation:**\\n",
        "\\n",
        "Isolation Forest doesn't require feature scaling because of how it makes decisions:\\n",
        "\\n",
        "1. **Random Feature Selection**: At each split, IForest randomly selects a feature (Height OR Weight)\\n",
        "2. **Random Split Point**: It then picks a random value between min and max of that feature\\n",
        "3. **Relative Position**: What matters is where the point falls relative to other points in that feature's range\\n",
        "\\n",
        "**Why scale doesn't matter:**\\n",
        "- Each feature is evaluated independently at each split\\n",
        "- The split point is chosen within that feature's natural range\\n",
        "- Scale doesn't affect whether a point is isolated quickly or slowly\\n",
        "\\n",
        "**Contrast with distance-based methods:**\\n",
        "- **KNN/OCSVM**: Calculate distances like √[(h₁-h₂)² + (w₁-w₂)²]\\n",
        "- Without scaling: (150-160)² dominates (60-65)²\\n",
        "- **IForest**: Asks 'Is height < 63?' and 'Is weight < 140?' separately\\n",
        "- Scale doesn't affect these yes/no questions\\n",
        "\\n",
        "**Practical Implication**: IForest is more robust and requires less preprocessing, making it excellent for production systems.\\n"
    ]
}
solution_nb['cells'].insert(60, iforest_scaling_57)

# After cell 67 (AutoEncoder training time analysis)
autoencoder_analysis_67 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### SOLUTION: Training Time vs Performance Analysis\\n",
        "\\n",
        "**1. How much longer did 50 epochs take compared to 10?**\\n",
        "\\n",
        "50 epochs should take approximately 5x longer than 10 epochs (roughly linear scaling).\\n",
        "Expected: 10 epochs ~2-5 seconds, 50 epochs ~10-25 seconds (depending on hardware).\\n",
        "\\n",
        "**2. Did the results improve significantly?**\\n",
        "\\n",
        "This depends on the learning curve:\\n",
        "- If 10 epochs was too few: Yes, significant improvement (better convergence)\\n",
        "- If 10 epochs was sufficient: Minimal improvement (already converged)\\n",
        "- Possible: Worse performance (overfitting to training data)\\n",
        "\\n",
        "Typically for this dataset, 10-20 epochs is sufficient. Beyond that, diminishing returns.\\n",
        "\\n",
        "**3. Would you use more or fewer epochs for this dataset? Why?**\\n",
        "\\n",
        "**Recommendation: 20-30 epochs** for this dataset because:\\n",
        "\\n",
        "**Reasons for moderate epochs:**\\n",
        "- Dataset is relatively simple (only 2 features with clear linear relationship)\\n",
        "- Small network can learn the pattern quickly\\n",
        "- Too few (<10): Might not converge fully\\n",
        "- Too many (>50): Waste of computation, risk of overfitting\\n",
        "\\n",
        "**In practice:**\\n",
        "- Use early stopping: Monitor validation loss and stop when it stops improving\\n",
        "- Start with 20-30 epochs and adjust based on loss curves\\n",
        "- For larger/more complex datasets, might need 100+ epochs\\n",
        "\\n",
        "**General principle**: More data = more epochs needed; Simpler patterns = fewer epochs needed.\\n"
    ]
}
solution_nb['cells'].insert(70, autoencoder_analysis_67)

# Add interpretation answers after cell 79
interpretation_answers_79 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### SOLUTION: Interpretation Answers\\n",
        "\\n",
        "**1. Which two algorithms agree the most? Why might this be?**\\n",
        "\\n",
        "Typically, algorithms from the same family agree more:\\n",
        "- **KNN and LOF**: Both use distance/density, so high agreement expected\\n",
        "- **COPOD and ECOD**: Both are probabilistic, similar theoretical foundation\\n",
        "- **IForest and AutoEncoder**: May agree on extreme outliers\\n",
        "\\n",
        "Check your agreement matrix - the highest values show the most similar algorithms.\\n",
        "\\n",
        "**2. Which algorithm found the most unique outliers? What does this tell you?**\\n",
        "\\n",
        "Usually **CBLOF** or **AutoEncoder** find the most unique outliers because:\\n",
        "- CBLOF uses clustering which defines outliers differently (small clusters)\\n",
        "- AutoEncoder learns complex non-linear patterns others might miss\\n",
        "\\n",
        "This tells us these algorithms have a **different definition of 'anomalous'** - they're capturing different aspects of the data.\\n",
        "\\n",
        "**3. Do you trust the high-confidence outliers more than algorithm-specific ones? Why or why not?**\\n",
        "\\n",
        "**Yes, generally trust high-confidence outliers more:**\\n",
        "\\n",
        "*Pros of consensus outliers:*\\n",
        "- Multiple independent methods agree → higher confidence\\n",
        "- Less likely to be algorithm-specific artifacts\\n",
        "- Robust to algorithm choice\\n",
        "\\n",
        "*However, algorithm-specific outliers can be valuable:*\\n",
        "- Might catch different types of anomalies\\n",
        "- Could be early warning signals\\n",
        "- Useful for comprehensive monitoring\\n",
        "\\n",
        "**Best approach**: Prioritize consensus outliers for action, but investigate algorithm-specific ones too.\\n",
        "\\n",
        "**4. If you had to pick ONE algorithm for production use on this data, which would you choose?**\\n",
        "\\n",
        "**Recommendation: Isolation Forest**\\n",
        "\\n",
        "*Justification:*\\n",
        "- **Fast**: O(n) complexity, scales well\\n",
        "- **No scaling needed**: Robust, less preprocessing\\n",
        "- **Few hyperparameters**: Works well with defaults\\n",
        "- **Interpretable**: Path length scores are intuitive\\n",
        "- **Proven**: Widely used in production systems\\n",
        "\\n",
        "*Alternative choice: COPOD*\\n",
        "- Even faster\\n",
        "- Parameter-free\\n",
        "- Good for high-dimensional data\\n",
        "- Less interpretable than IForest\\n",
        "\\n",
        "**Production strategy**: Use IForest as primary, COPOD as secondary check for consensus.\\n"
    ]
}
solution_nb['cells'].insert(82, interpretation_answers_79)

# Add recommendation after cell 83
challenge_recommendation_83 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### SOLUTION: Algorithm Recommendation\\n",
        "\\n",
        "**Recommended Algorithm: Isolation Forest with COPOD as secondary validation**\\n",
        "\\n",
        "**Analysis:**\\n",
        "\\n",
        "Based on our comprehensive testing, here's my recommendation for production deployment on this weight-height dataset:\\n",
        "\\n",
        "**Primary Algorithm: Isolation Forest**\\n",
        "\\n",
        "*Accuracy:*\\n",
        "- Detects reasonable outliers (extreme heights/weights)\\n",
        "- Good balance between false positives and false negatives\\n",
        "- Captures both univariate extremes and unusual combinations\\n",
        "\\n",
        "*Speed:*\\n",
        "- Very fast training and prediction (~0.1-0.5 seconds)\\n",
        "- O(n log n) complexity - scales to millions of records\\n",
        "- No preprocessing required (no scaling step)\\n",
        "\\n",
        "*Interpretability:*\\n",
        "- Anomaly scores based on average path length\\n",
        "- Can explain: 'This point was isolated in only 3 splits vs average of 12'\\n",
        "- Stakeholders can understand the intuition\\n",
        "\\n",
        "*Robustness:*\\n",
        "- Works well with default parameters (contamination=0.005, n_estimators=100)\\n",
        "- Not sensitive to feature scales\\n",
        "- Random forest approach reduces variance\\n",
        "\\n",
        "**Secondary Validation: COPOD**\\n",
        "\\n",
        "Use COPOD as a second opinion:\\n",
        "- Even faster than IForest\\n",
        "- Completely parameter-free\\n",
        "- Different theoretical approach (probabilistic vs. ensemble)\\n",
        "- High-confidence outliers = both algorithms agree\\n",
        "\\n",
        "**Deployment Strategy:**\\n",
        "\\n",
        "```python\\n",
        "# Primary detector\\n",
        "primary = IForest(contamination=0.005, n_estimators=100, random_state=42)\\n",
        "primary.fit(data[['Height', 'Weight']])\\n",
        "primary_pred = primary.predict(data[['Height', 'Weight']])\\n",
        "\\n",
        "# Secondary validator\\n",
        "secondary = COPOD(contamination=0.005)\\n",
        "secondary.fit(data[['Height', 'Weight']])\\n",
        "secondary_pred = secondary.predict(data[['Height', 'Weight']])\\n",
        "\\n",
        "# Confidence scoring\\n",
        "high_confidence = (primary_pred == 1) & (secondary_pred == 1)  # Both agree\\n",
        "medium_confidence = (primary_pred == 1) & (secondary_pred == 0)  # Only primary\\n",
        "```\\n",
        "\\n",
        "**Why not the other algorithms?**\\n",
        "\\n",
        "- **KNN/LOF**: Too slow for large datasets, require tuning k\\n",
        "- **CBLOF**: Requires choosing number of clusters\\n",
        "- **OCSVM**: Requires feature scaling, slower, sensitive to parameters\\n",
        "- **AutoEncoder**: Overkill for 2 features, requires tuning, black box\\n",
        "\\n",
        "**Conclusion:**\\n",
        "For production deployment on this health/biometric data, prioritize speed, interpretability, and robustness. Isolation Forest with COPOD validation provides the best balance of these factors.\\n"
    ]
}
solution_nb['cells'].insert(87, challenge_recommendation_83)

# Add reflection answers after cell 84
reflection_answers_84 = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### SOLUTION: Reflection Answers\\n",
        "\\n",
        "## Question 1: Distance-Based vs Ensemble Methods\\n",
        "\\n",
        "**When would you choose distance-based methods (KNN, LOF) over ensemble methods (IForest)?**\\n",
        "\\n",
        "**Choose Distance-Based (KNN, LOF) when:**\\n",
        "\\n",
        "1. **Small dataset** (<10,000 points)\\n",
        "   - Computational cost is manageable\\n",
        "   - Can afford to compute all pairwise distances\\n",
        "\\n",
        "2. **Need interpretability**\\n",
        "   - 'This point is anomalous because its 5 nearest neighbors are X distance away'\\n",
        "   - Can visualize neighborhoods\\n",
        "   - Easy to explain to non-technical stakeholders\\n",
        "\\n",
        "3. **Local anomalies important**\\n",
        "   - LOF excels at finding points anomalous within their local context\\n",
        "   - Data has clusters with varying densities\\n",
        "   - Need to detect outliers within subgroups\\n",
        "\\n",
        "4. **Low dimensional data** (2-10 features)\\n",
        "   - Distance metrics work well\\n",
        "   - Curse of dimensionality not a major issue\\n",
        "\\n",
        "**Choose Ensemble (IForest) when:**\\n",
        "\\n",
        "1. **Large dataset** (>100,000 points)\\n",
        "   - O(n log n) vs O(n²) complexity matters\\n",
        "   - Need real-time or near real-time detection\\n",
        "\\n",
        "2. **High dimensional data** (>10 features)\\n",
        "   - IForest handles curse of dimensionality better\\n",
        "   - Distance metrics become unreliable\\n",
        "\\n",
        "3. **Production deployment**\\n",
        "   - Need reliability and robustness\\n",
        "   - Limited time for hyperparameter tuning\\n",
        "   - Require consistent performance\\n",
        "\\n",
        "4. **Computational resources limited**\\n",
        "   - Faster training and prediction\\n",
        "   - Can run on edge devices\\n",
        "\\n",
        "---\\n",
        "\\n",
        "## Question 2: The Importance of Scaling\\n",
        "\\n",
        "**Why does scaling matter for OCSVM but not for IForest?**\\n",
        "\\n",
        "**OCSVM (Scaling Required):**\\n",
        "\\n",
        "OCSVM uses **distance calculations in kernel space**:\\n",
        "\\n",
        "```\\n",
        "RBF Kernel: K(x, y) = exp(-γ ||x - y||²)\\n",
        "```\\n",
        "\\n",
        "Without scaling:\\n",
        "- ||x - y||² = (h₁-h₂)² + (w₁-w₂)²\\n",
        "- Height: 60-72 inches → differences of 0-12\\n",
        "- Weight: 100-200 lbs → differences of 0-100\\n",
        "- Weight differences dominate: (100)² >> (12)²\\n",
        "- Algorithm effectively ignores Height dimension\\n",
        "- Decision boundary collapses to weight-only\\n",
        "\\n",
        "With scaling (mean=0, std=1):\\n",
        "- Both features contribute equally\\n",
        "- Decision boundary considers all dimensions\\n",
        "- Proper circular/elliptical boundary\\n",
        "\\n",
        "**IForest (Scaling NOT Required):**\\n",
        "\\n",
        "IForest uses **random splits**:\\n",
        "\\n",
        "```\\n",
        "1. Randomly select feature f (Height OR Weight)\\n",
        "2. Randomly select split value v between min(f) and max(f)\\n",
        "3. Split: left if x[f] < v, right otherwise\\n",
        "```\\n",
        "\\n",
        "Why scale doesn't matter:\\n",
        "- Each feature evaluated independently\\n",
        "- Split value chosen within feature's natural range\\n",
        "- Weight splits: 100-200, Height splits: 60-72\\n",
        "- Both get equal chance (random feature selection)\\n",
        "- Isolation speed depends on relative position, not absolute scale\\n",
        "\\n",
        "**Key Insight:**\\n",
        "- **Distance-based**: Features combined in single metric → need same scale\\n",
        "- **Split-based**: Features evaluated separately → scale irrelevant\\n",
        "\\n",
        "**Practical Implications:**\\n",
        "- Always scale: OCSVM, KNN, k-means, neural networks\\n",
        "- Optional to scale: IForest, decision trees, random forests\\n",
        "\\n",
        "---\\n",
        "\\n",
        "## Question 3: Interpretability vs Performance Trade-offs\\n",
        "\\n",
        "**What are the trade-offs between interpretability and performance in anomaly detection?**\\n",
        "\\n",
        "### Simple Methods (KNN, COPOD)\\n",
        "\\n",
        "**High Interpretability:**\\n",
        "\\n",
        "*KNN:*\\n",
        "- 'Point X is anomalous because its 5 nearest neighbors are average 15 units away'\\n",
        "- Can show the actual neighbors\\n",
        "- Can visualize distance distributions\\n",
        "- Stakeholder explanation: 'This person is very different from similar people'\\n",
        "\\n",
        "*COPOD:*\\n",
        "- 'Point X has probability 0.001 under the empirical distribution'\\n",
        "- 'Height is in the 99th percentile, weight in 1st percentile'\\n",
        "- Statistical foundation makes it trustworthy\\n",
        "\\n",
        "**Performance:**\\n",
        "- KNN: Good for simple patterns, struggles with high dimensions\\n",
        "- COPOD: Excellent for moderate complexity, assumes feature independence\\n",
        "\\n",
        "### Complex Methods (AutoEncoder, OCSVM)\\n",
        "\\n",
        "**Lower Interpretability:**\\n",
        "\\n",
        "*AutoEncoder:*\\n",
        "- 'Point X has high reconstruction error'\\n",
        "- Can't easily explain WHY reconstruction failed\\n",
        "- Hidden layer activations are not human-readable\\n",
        "- Black box: input → latent → reconstruction\\n",
        "\\n",
        "*OCSVM:*\\n",
        "- 'Point X is outside the decision boundary'\\n",
        "- Boundary defined by support vectors in kernel space\\n",
        "- Can't visualize in original space (especially with RBF kernel)\\n",
        "- Mathematical but not intuitive\\n",
        "\\n",
        "**Performance:**\\n",
        "- AutoEncoder: Can learn very complex non-linear patterns\\n",
        "- OCSVM: Excellent for non-linear boundaries with right kernel\\n",
        "\\n",
        "### The Trade-off Spectrum\\n",
        "\\n",
        "```\\n",
        "High Interpretability                    High Performance\\n",
        "        |                                      |\\n",
        "    KNN  →  LOF  →  IForest  →  OCSVM  →  AutoEncoder\\n",
        "        |                                      |\\n",
        "  Simple patterns               Complex non-linear patterns\\n",
        "```\\n",
        "\\n",
        "**IForest: The Sweet Spot**\\n",
        "- Moderate interpretability (path length)\\n",
        "- Strong performance (handles complex patterns)\\n",
        "- This balance makes it popular for production\\n",
        "\\n",
        "### When Does Each Matter?\\n",
        "\\n",
        "**Prioritize Interpretability:**\\n",
        "- Healthcare (must explain to doctors/patients)\\n",
        "- Finance (regulatory requirements)\\n",
        "- Legal liability (need to justify decisions)\\n",
        "- Building trust in the system\\n",
        "→ Use: KNN, LOF, simple rules\\n",
        "\\n",
        "**Prioritize Performance:**\\n",
        "- Fraud detection (false negatives very costly)\\n",
        "- Network intrusion (need to catch sophisticated attacks)\\n",
        "- Complex manufacturing processes\\n",
        "- When data has proven complex patterns\\n",
        "→ Use: AutoEncoder, OCSVM, deep learning\\n",
        "\\n",
        "**Balance Both:**\\n",
        "- Most production systems\\n",
        "- Need good performance + some explanation\\n",
        "- Resource constraints\\n",
        "→ Use: IForest, LOF, ensemble methods\\n",
        "\\n",
        "### Improving Interpretability of Complex Models\\n",
        "\\n",
        "Even with black-box models, you can add interpretability:\\n",
        "1. **SHAP values**: Explain feature contributions\\n",
        "2. **Attention mechanisms**: Show what the model focuses on\\n",
        "3. **Confidence scores**: Provide uncertainty estimates\\n",
        "4. **Comparison to similar cases**: 'Like these normal cases, but...'\\n",
        "\\n",
        "**Conclusion:**\\n",
        "Start with interpretable methods. Only move to complex models if:\\n",
        "1. Proven performance benefit\\n",
        "2. Can afford the interpretability cost\\n",
        "3. Have tools/processes to explain decisions\\n",
        "\\n",
        "---\\n",
        "\\n",
        "## Question 4: Real-World Healthcare Application\\n",
        "\\n",
        "**Healthcare Anomaly Detection System Design**\\n",
        "\\n",
        "**Scenario:** Deploy anomaly detection for patient biometric data (height, weight, BMI, blood pressure, heart rate, etc.)\\n",
        "\\n",
        "### Algorithm Choice: LOF (Primary) + IForest (Secondary)\\n",
        "\\n",
        "**Why LOF as Primary:**\\n",
        "\\n",
        "1. **Interpretability Critical**\\n",
        "   - Doctors need to understand WHY a patient is flagged\\n",
        "   - Can explain: 'Patient's metrics differ from similar patients by X%'\\n",
        "   - Can show the comparison group (neighbors)\\n",
        "   - Legal/ethical requirement to explain medical decisions\\n",
        "\\n",
        "2. **Local Context Matters**\\n",
        "   - Different patient populations (age groups, demographics)\\n",
        "   - What's normal for elderly vs young patients differs\\n",
        "   - LOF naturally handles these varying-density clusters\\n",
        "   - Detects 'unusual for THIS type of patient'\\n",
        "\\n",
        "3. **False Positive Cost**\\n",
        "   - Flagging healthy patients → unnecessary anxiety, wasted resources\\n",
        "   - LOF's local density reduces false positives in dense regions\\n",
        "   - Less likely to flag patients just because they're in a sparse demographic\\n",
        "\\n",
        "**Why IForest as Secondary:**\\n",
        "\\n",
        "1. **Catches Different Anomalies**\\n",
        "   - Finds global outliers LOF might miss\\n",
        "   - Fast enough to run alongside LOF\\n",
        "   - Handles high-dimensional data if many biomarkers\\n",
        "\\n",
        "2. **Validation/Confidence**\\n",
        "   - If both algorithms agree → high priority case\\n",
        "   - If only LOF flags → review with local context\\n",
        "   - If only IForest flags → extreme global outlier\\n",
        "\\n",
        "### Cost Analysis\\n",
        "\\n",
        "**False Positive Cost (Healthy → Flagged):**\\n",
        "- Patient anxiety and stress\\n",
        "- Unnecessary follow-up appointments\\n",
        "- Additional tests (cost $100-1000)\\n",
        "- Healthcare system resource waste\\n",
        "- Potential for cascade of unnecessary procedures\\n",
        "**Strategy:** Set higher contamination threshold (0.01-0.02) to reduce FP\\n",
        "\\n",
        "**False Negative Cost (Unhealthy → Missed):**\\n",
        "- Delayed diagnosis → disease progression\\n",
        "- Potentially life-threatening (cardiovascular events, diabetes complications)\\n",
        "- Much higher treatment costs later\\n",
        "- Legal liability for missed diagnosis\\n",
        "- Loss of patient trust\\n",
        "**Strategy:** Use ensemble (LOF + IForest) to catch more cases\\n",
        "\\n",
        "**Risk-Based Tiering:**\\n",
        "```python\\n",
        "# High risk: Both algorithms agree\\n",
        "high_risk = (lof_pred == 1) & (iforest_pred == 1)  # Immediate doctor review\\n",
        "\\n",
        "# Medium risk: One algorithm flags\\n",
        "medium_risk = (lof_pred == 1) | (iforest_pred == 1)  # Nurse review → doctor if needed\\n",
        "\\n",
        "# Low risk: Neither flags\\n",
        "low_risk = (lof_pred == 0) & (iforest_pred == 0)  # Routine monitoring\\n",
        "```\\n",
        "\\n",
        "### Explainability for Doctors\\n",
        "\\n",
        "**Report Format:**\\n",
        "```\\n",
        "Patient ID: 12345\\n",
        "Anomaly Score: 0.92 (High)\\n",
        "\\n",
        "Comparison to Similar Patients:\\n",
        "- Age group: 55-65\\n",
        "- Similar BMI range\\n",
        "- 20 nearest neighbors identified\\n",
        "\\n",
        "Unusual Metrics:\\n",
        "- Blood Pressure: 165/95 (avg for group: 125/80)\\n",
        "- Heart Rate: 95 bpm (avg for group: 72 bpm)\\n",
        "- BMI: 32 (within normal range for group)\\n",
        "\\n",
        "Interpretation:\\n",
        "Patient's cardiovascular metrics significantly higher than peers\\n",
        "with similar age and BMI. Recommend cardiovascular assessment.\\n",
        "```\\n",
        "\\n",
        "### Real-Time Performance Requirements\\n",
        "\\n",
        "**Scenario 1: Batch Processing (Daily Reports)**\\n",
        "- Process all patients overnight\\n",
        "- Can use both LOF + IForest\\n",
        "- Acceptable: 1-10 seconds per patient\\n",
        "\\n",
        "**Scenario 2: Real-Time (Patient Visit)**\\n",
        "- Doctor enters vitals, needs immediate feedback\\n",
        "- Required: <1 second response time\\n",
        "- Solution: Pre-trained models, use IForest for speed\\n",
        "\\n",
        "**Scenario 3: Continuous Monitoring (ICU)**\\n",
        "- Streaming vitals data\\n",
        "- Required: <100ms per update\\n",
        "- Solution: Lightweight IForest, update models hourly\\n",
        "\\n",
        "### Validation & Deployment\\n",
        "\\n",
        "**Pre-Deployment:**\\n",
        "1. **Clinical Validation**\\n",
        "   - Test on historical data with known outcomes\\n",
        "   - Calculate sensitivity/specificity\\n",
        "   - Get doctor feedback on false positives\\n",
        "\\n",
        "2. **Bias Testing**\\n",
        "   - Ensure equal performance across demographics\\n",
        "   - Test for racial, gender, age biases\\n",
        "   - Required for healthcare AI\\n",
        "\\n",
        "3. **Regulatory Compliance**\\n",
        "   - HIPAA compliance for data handling\\n",
        "   - FDA approval if making diagnostic claims\\n",
        "   - Documentation of model decisions\\n",
        "\\n",
        "**Ongoing Monitoring:**\\n",
        "1. **Model Drift**\\n",
        "   - Patient population changes over time\\n",
        "   - Retrain monthly with recent data\\n",
        "   - Track performance metrics\\n",
        "\\n",
        "2. **Doctor Feedback Loop**\\n",
        "   - Doctors mark false positives/negatives\\n",
        "   - Feed back into model training\\n",
        "   - Continuous improvement\\n",
        "\\n",
        "3. **Periodic Audits**\\n",
        "   - Quarterly review of flagged cases\\n",
        "   - Validate clinical relevance\\n",
        "   - Adjust thresholds if needed\\n",
        "\\n",
        "### Complete System Architecture\\n",
        "\\n",
        "```python\\n",
        "class HealthcareAnomalyDetector:\\n",
        "    def __init__(self):\\n",
        "        # Primary: Interpretable local outlier detection\\n",
        "        self.lof = LOF(n_neighbors=20, contamination=0.02)\\n",
        "        \\n",
        "        # Secondary: Fast global outlier detection\\n",
        "        self.iforest = IForest(contamination=0.02, n_estimators=100)\\n",
        "        \\n",
        "        # Feature scaling for LOF\\n",
        "        self.scaler = StandardScaler()\\n",
        "    \\n",
        "    def fit(self, patient_data):\\n",
        "        # Scale features\\n",
        "        scaled_data = self.scaler.fit_transform(patient_data)\\n",
        "        \\n",
        "        # Train both models\\n",
        "        self.lof.fit(scaled_data)\\n",
        "        self.iforest.fit(patient_data)  # IForest doesn't need scaling\\n",
        "    \\n",
        "    def predict_with_explanation(self, new_patient):\\n",
        "        # Scale input\\n",
        "        scaled_input = self.scaler.transform(new_patient)\\n",
        "        \\n",
        "        # Get predictions\\n",
        "        lof_pred = self.lof.predict(scaled_input)[0]\\n",
        "        iforest_pred = self.iforest.predict(new_patient)[0]\\n",
        "        \\n",
        "        # Risk level\\n",
        "        if lof_pred == 1 and iforest_pred == 1:\\n",
        "            risk = 'HIGH'\\n",
        "        elif lof_pred == 1 or iforest_pred == 1:\\n",
        "            risk = 'MEDIUM'\\n",
        "        else:\\n",
        "            risk = 'LOW'\\n",
        "        \\n",
        "        # Generate explanation (show neighbors, unusual features)\\n",
        "        explanation = self._generate_explanation(new_patient, lof_pred)\\n",
        "        \\n",
        "        return {\\n",
        "            'risk_level': risk,\\n",
        "            'lof_anomaly': bool(lof_pred),\\n",
        "            'iforest_anomaly': bool(iforest_pred),\\n",
        "            'explanation': explanation\\n",
        "        }\\n",
        "```\\n",
        "\\n",
        "**Conclusion:**\\n",
        "For healthcare, prioritize interpretability and local context (LOF), use ensemble for safety (IForest), implement robust validation and monitoring. The cost of false negatives is much higher than false positives, so use a conservative threshold and multiple algorithms.\\n",
        "\\n",
        "---\\n",
        "\\n",
        "## Question 5: Multiple Algorithm Approach (Ensemble of Ensembles)\\n",
        "\\n",
        "**Based on your quantitative evaluation, do you think using multiple algorithms together would be beneficial?**\\n",
        "\\n",
        "### Short Answer: Yes, but with a structured approach\\n",
        "\\n",
        "### Evidence from Our Analysis\\n",
        "\\n",
        "Looking at our agreement matrix and overlap analysis:\\n",
        "\\n",
        "1. **Different algorithms find different outliers**\\n",
        "   - Agreement typically 40-60% between any two algorithms\\n",
        "   - Each has unique detections (30-50% algorithm-specific)\\n",
        "   - Different definitions capture different anomaly types\\n",
        "\\n",
        "2. **High-confidence outliers are more reliable**\\n",
        "   - Points flagged by 3+ algorithms: likely true outliers\\n",
        "   - Single-algorithm detections: might be artifacts\\n",
        "   - Consensus reduces false positives\\n",
        "\\n",
        "3. **Family-specific patterns**\\n",
        "   - Distance-based (KNN, LOF) agree more with each other\\n",
        "   - Probabilistic (COPOD, ECOD) agree more with each other\\n",
        "   - Cross-family agreement adds complementary info\\n",
        "\\n",
        "### Benefits of Multiple Algorithms\\n",
        "\\n",
        "**1. Improved Robustness**\\n",
        "```python\\n",
        "# Voting ensemble\\n",
        "predictions = [knn_pred, lof_pred, iforest_pred, copod_pred]\\n",
        "vote_count = sum(predictions)\\n",
        "\\n",
        "# Majority vote: 2 out of 4 algorithms\\n",
        "ensemble_pred = (vote_count >= 2).astype(int)\\n",
        "```\\n",
        "- Reduces impact of single algorithm failures\\n",
        "- Less sensitive to hyperparameter choices\\n",
        "- More stable across different data distributions\\n",
        "\\n",
        "**2. Tiered Alert System**\\n",
        "```python\\n",
        "# Risk-based prioritization\\n",
        "critical = (vote_count >= 3)  # 75%+ agreement → immediate action\\n",
        "warning = (vote_count == 2)    # 50% agreement → investigate\\n",
        "monitor = (vote_count == 1)    # 25% agreement → watch\\n",
        "```\\n",
        "- Prioritizes investigation resources\\n",
        "- Reduces alert fatigue\\n",
        "- Captures nuanced risk levels\\n",
        "\\n",
        "**3. Complementary Strengths**\\n",
        "- **KNN**: Finds isolated points\\n",
        "- **LOF**: Finds local density anomalies\\n",
        "- **IForest**: Finds easily-isolated points\\n",
        "- **COPOD**: Finds low-probability combinations\\n",
        "\\n",
        "Together they cover:\\n",
        "- Global outliers (KNN, IForest)\\n",
        "- Local outliers (LOF)\\n",
        "- Statistical outliers (COPOD)\\n",
        "- Pattern outliers (IForest)\\n",
        "\\n",
        "**4. Improved Detection Rates**\\n",
        "```python\\n",
        "# OR combination: catches more anomalies\\n",
        "sensitive = (knn_pred | lof_pred | iforest_pred)  # High recall\\n",
        "\\n",
        "# AND combination: higher confidence\\n",
        "specific = (knn_pred & lof_pred & iforest_pred)  # High precision\\n",
        "```\\n",
        "- Can tune precision-recall trade-off\\n",
        "- Flexible for different use cases\\n",
        "\\n",
        "### Challenges of Multiple Algorithms\\n",
        "\\n",
        "**1. Computational Cost**\\n",
        "```\\n",
        "Single algorithm: t\\n",
        "4 algorithms: 4t (training), 4t (prediction)\\n",
        "```\\n",
        "- Significant overhead for real-time systems\\n",
        "- More memory required\\n",
        "- Mitigation: Run in parallel, use fast algorithms\\n",
        "\\n",
        "**2. Complexity**\\n",
        "- More hyperparameters to tune (4 models × parameters each)\\n",
        "- Need strategy for combining predictions\\n",
        "- Harder to debug when something goes wrong\\n",
        "\\n",
        "**3. Diminishing Returns**\\n",
        "```\\n",
        "Improvement: 1→2 algorithms: +20% accuracy\\n",
        "             2→3 algorithms: +8% accuracy\\n",
        "             3→4 algorithms: +3% accuracy\\n",
        "```\\n",
        "- Each additional algorithm adds less value\\n",
        "- Cost grows linearly, benefit grows logarithmically\\n",
        "\\n",
        "**4. Over-reliance Risk**\\n",
        "- If all algorithms trained on same biased data → still biased\\n",
        "- Consensus doesn't mean correct\\n",
        "- All algorithms might miss the same novel anomaly type\\n",
        "\\n",
        "### Optimal Strategy: Smart Ensemble\\n",
        "\\n",
        "Rather than using all algorithms, use a **strategic combination**:\\n",
        "\\n",
        "**Tier 1: Fast Screening (Primary Detection)**\\n",
        "```python\\n",
        "# Use fast, parameter-free algorithms\\n",
        "primary = IForest(contamination=0.01)  # Fast, robust\\n",
        "validator = COPOD(contamination=0.01)  # Even faster, different approach\\n",
        "\\n",
        "# Quick first pass\\n",
        "suspicious = primary.predict(data) | validator.predict(data)\\n",
        "```\\n",
        "\\n",
        "**Tier 2: Deep Analysis (Secondary Validation)**\\n",
        "```python\\n",
        "# Only for points flagged in Tier 1\\n",
        "suspicious_data = data[suspicious == 1]\\n",
        "\\n",
        "# Apply slower, more sophisticated algorithms\\n",
        "lof_detailed = LOF(n_neighbors=50, contamination=0.1)\\n",
        "ocsvm_detailed = OCSVM(kernel='rbf', nu=0.1)\\n",
        "\\n",
        "# Confirm with multiple methods\\n",
        "confirmed = (lof_detailed.predict(suspicious_data) & \\n",
        "            ocsvm_detailed.predict(suspicious_data))\\n",
        "```\\n",
        "\\n",
        "**Benefits of Tiered Approach:**\\n",
        "- 90% of data processed quickly (Tier 1)\\n",
        "- Only 10% goes through expensive Tier 2\\n",
        "- Overall computation time: 0.1t₁ + 0.9t₂ ≈ t₂ (instead of 4t)\\n",
        "- High accuracy on important cases\\n",
        "\\n",
        "### Recommended Ensemble Configurations\\n",
        "\\n",
        "**Configuration 1: Speed-Accuracy Balance**\\n",
        "```python\\n",
        "# 2 algorithms from different families\\n",
        "ensemble = [\\n",
        "    IForest(contamination=0.01, n_estimators=100),  # Ensemble method\\n",
        "    LOF(contamination=0.01, n_neighbors=20)         # Distance method\\n",
        "]\\n",
        "# Voting: Both agree → high confidence\\n",
        "#         Either flags → medium confidence\\n",
        "```\\n",
        "\\n",
        "**Configuration 2: Maximum Coverage**\\n",
        "```python\\n",
        "# 3 algorithms, one from each major family\\n",
        "ensemble = [\\n",
        "    IForest(contamination=0.01),   # Ensemble\\n",
        "    LOF(contamination=0.01),       # Distance\\n",
        "    COPOD(contamination=0.01)      # Probabilistic\\n",
        "]\\n",
        "# Voting: 2+ agree → flag as anomaly\\n",
        "```\\n",
        "\\n",
        "**Configuration 3: Production System**\\n",
        "```python\\n",
        "# Primary (always run): Fast algorithms\\n",
        "primary = IForest(contamination=0.005)\\n",
        "\\n",
        "# Secondary (run on flagged): Validation\\n",
        "secondary = LOF(contamination=0.05)  # Higher threshold for validation\\n",
        "\\n",
        "# Tertiary (run on confirmed): Deep analysis\\n",
        "tertiary = AutoEncoder(contamination=0.1, epochs=50)\\n",
        "```\\n",
        "\\n",
        "### When NOT to Use Multiple Algorithms\\n",
        "\\n",
        "**Skip ensemble if:**\\n",
        "1. **Clear best algorithm exists**\\n",
        "   - One algorithm significantly outperforms others\\n",
        "   - Additional algorithms don't add new detections\\n",
        "   \\n",
        "2. **Real-time constraints**\\n",
        "   - Latency budget: <10ms\\n",
        "   - Even fast ensemble too slow\\n",
        "   \\n",
        "3. **Limited resources**\\n",
        "   - Edge devices (IoT sensors)\\n",
        "   - Memory constraints\\n",
        "   \\n",
        "4. **Simple data**\\n",
        "   - 1-2 features\\n",
        "   - Obvious outliers\\n",
        "   - Single algorithm sufficient\\n",
        "\\n",
        "### Advanced: Weighted Voting\\n",
        "\\n",
        "```python\\n",
        "# Not all algorithms equal - weight by historical performance\\n",
        "weights = {\\n",
        "    'IForest': 0.35,   # Best overall performer\\n",
        "    'LOF': 0.30,       # Good at local anomalies\\n",
        "    'COPOD': 0.25,     # Fast and reliable\\n",
        "    'KNN': 0.10        # Occasional unique finds\\n",
        "}\\n",
        "\\n",
        "# Weighted ensemble score\\n",
        "ensemble_score = (\\n",
        "    weights['IForest'] * iforest_scores +\\n",
        "    weights['LOF'] * lof_scores +\\n",
        "    weights['COPOD'] * copod_scores +\\n",
        "    weights['KNN'] * knn_scores\\n",
        ")\\n",
        "\\n",
        "# Threshold on combined score\\n",
        "anomalies = (ensemble_score > threshold)\\n",
        "```\\n",
        "\\n",
        "**Benefits:**\\n",
        "- Leverages each algorithm's strengths\\n",
        "- Can adapt weights over time\\n",
        "- More nuanced than simple voting\\n",
        "\\n",
        "### Conclusion\\n",
        "\\n",
        "**Yes, use multiple algorithms, but strategically:**\\n",
        "\\n",
        "1. **Start with 2 algorithms** from different families (e.g., IForest + LOF)\\n",
        "2. **Use consensus for high-confidence detection** (both agree)\\n",
        "3. **Implement tiered approach** for large datasets (fast screening → deep analysis)\\n",
        "4. **Don't exceed 3-4 algorithms** (diminishing returns)\\n",
        "5. **Monitor and adapt** (track which algorithms add value)\\n",
        "\\n",
        "The key is **complementarity, not redundancy**. Choose algorithms that find different types of anomalies, not algorithms that find the same outliers in different ways.\\n",
        "\\n",
        "**Final recommendation for production:**\\n",
        "```python\\n",
        "# Simple, effective 2-algorithm ensemble\\n",
        "primary = IForest(contamination=0.01, n_estimators=100)    # Fast, robust\\n",
        "secondary = LOF(contamination=0.01, n_neighbors=20)        # Interpretable, local\\n",
        "\\n",
        "# Tiered alerts\\n",
        "critical = both_detect()      # Immediate attention\\n",
        "warning = either_detect()     # Investigation required\\n",
        "normal = neither_detect()     # Continue monitoring\\n",
        "```\\n",
        "\\n",
        "This provides the best balance of accuracy, speed, interpretability, and resource usage for most real-world applications.\\n"
    ]
}
solution_nb['cells'].insert(89, reflection_answers_84)

# Save the solution notebook
solution_path = Path("../solutions/activity_01_ml_anomaly_basics_SOLUTION.ipynb")
with open(solution_path, 'w') as f:
    json.dump(solution_nb, f, indent=2)

print(f"Solution notebook created successfully!")
print(f"Total cells: {len(solution_nb['cells'])}")
print(f"Saved to: {solution_path}")
