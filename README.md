# Advanced Machine Learning Class Outline

> [!NOTE]
> - [X] **Content for Day 1: Advanced Data Management and Model Optimization**
> - [X] **Content for Day 2: Ensemble Methods, and Model Robustness**
> - [X] **Content for Day 3: Model Explainability, and Deployment Considerations**

## Environment Setup

Below are instructions for setting up virtual environments using different tools.

### Using uv

1. Install uv if not already installed:
   ```bash
   pip install uv
   ```

2. Create a virtual environment:
   ```bash
   uv venv dev1 --python=3.12
   ```

3. Activate the environment:
   ```bash
   source dev1/bin/activate  # On macOS/Linux
   # or
   dev1\Scripts\activate     # On Windows
   ```

4. Deactivate the environment:
   ```bash
   deactivate
   ```

### Using venv (Python built-in)

1. Create a virtual environment:
   ```bash
   python3.12 -m venv dev1
   ```

2. Activate the environment:
   ```bash
   source dev1/bin/activate  # On macOS/Linux
   # or
   dev1\Scripts\activate     # On Windows
   ```

3. Deactivate the environment:
   ```bash
   deactivate
   ```

### Using conda

1. Create a conda environment:
   ```bash
   conda create -n dev1 python=3.12
   ```

2. Activate the environment:
   ```bash
   conda activate dev1
   ```

3. Deactivate the environment:
   ```bash
   conda deactivate
   ```

## Getting Started with Git

### Cloning the Repository

To get a copy of this repository on your local machine:

```bash
git clone https://github.com/tatwan/adv_ml_ds.git
cd adv_ml_ds
```

### Updating the Repository

To update your local copy with the latest changes from the remote repository:

```bash
git pull origin main
```

### Handling Modifications and Conflicts

If you've made local modifications and want to update:

1. **Commit your changes first** (if you want to keep them):
   ```bash
   git add .
   git commit -m "Your commit message"
   git pull origin main
   ```

2. **Stash your changes** (if you want to temporarily save them):
   ```bash
   git stash
   git pull origin main
   git stash pop  # To restore your changes
   ```

3. **If conflicts occur during pull**:
   - Git will notify you of conflicts
   - Edit the conflicted files to resolve conflicts
   - Stage the resolved files:
     ```bash
     git add <resolved_file>
     ```
   - Complete the merge:
     ```bash
     git commit
     ```

4. **Force update** (use with caution, this will overwrite local changes):
   ```bash
   git reset --hard origin/main
   ```
