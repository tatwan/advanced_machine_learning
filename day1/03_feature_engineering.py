"""
Day 1 - Module 3: Feature Engineering
Topic: Advanced Feature Engineering Techniques

This module covers:
- Creating new features from existing ones
- Polynomial features and interactions
- Encoding categorical variables
- Feature extraction and transformation
- Domain-specific feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    PolynomialFeatures, 
    StandardScaler,
    OneHotEncoder,
    LabelEncoder,
    OrdinalEncoder
)
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    mutual_info_classif,
    RFE
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from category_encoders import TargetEncoder, WOEEncoder
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Comprehensive feature engineering toolkit.
    """
    
    def __init__(self, data):
        """
        Initialize the FeatureEngineer.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data for feature engineering
        """
        self.data = data.copy()
        self.feature_history = []
    
    def create_polynomial_features(self, columns, degree=2, interaction_only=False):
        """
        Create polynomial features.
        
        Parameters:
        -----------
        columns : list
            Columns to create polynomial features from
        degree : int, default=2
            Degree of polynomial features
        interaction_only : bool, default=False
            If True, only interaction features are produced
        
        Returns:
        --------
        self : FeatureEngineer
            Returns self for method chaining
        """
        poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
        poly_features = poly.fit_transform(self.data[columns])
        
        # Create column names
        feature_names = poly.get_feature_names_out(columns)
        
        # Add to dataframe
        for i, name in enumerate(feature_names):
            if name not in columns:  # Don't duplicate original features
                self.data[name] = poly_features[:, i]
        
        self.feature_history.append(f"Created polynomial features (degree={degree}) for {columns}")
        return self
    
    def create_interactions(self, column_pairs):
        """
        Create interaction features between specific column pairs.
        
        Parameters:
        -----------
        column_pairs : list of tuples
            List of column pairs to create interactions for
        
        Returns:
        --------
        self : FeatureEngineer
            Returns self for method chaining
        """
        for col1, col2 in column_pairs:
            interaction_name = f"{col1}_x_{col2}"
            self.data[interaction_name] = self.data[col1] * self.data[col2]
            self.feature_history.append(f"Created interaction: {interaction_name}")
        
        return self
    
    def create_ratio_features(self, numerator_cols, denominator_cols):
        """
        Create ratio features.
        
        Parameters:
        -----------
        numerator_cols : list
            Columns to use as numerators
        denominator_cols : list
            Columns to use as denominators
        
        Returns:
        --------
        self : FeatureEngineer
            Returns self for method chaining
        """
        for num_col in numerator_cols:
            for den_col in denominator_cols:
                if num_col != den_col:
                    ratio_name = f"{num_col}_div_{den_col}"
                    # Add small epsilon to avoid division by zero
                    self.data[ratio_name] = self.data[num_col] / (self.data[den_col] + 1e-8)
                    self.feature_history.append(f"Created ratio: {ratio_name}")
        
        return self
    
    def create_aggregate_features(self, group_by_col, agg_cols, agg_funcs=['mean', 'std', 'min', 'max']):
        """
        Create aggregate features based on groupings.
        
        Parameters:
        -----------
        group_by_col : str
            Column to group by
        agg_cols : list
            Columns to aggregate
        agg_funcs : list, default=['mean', 'std', 'min', 'max']
            Aggregation functions to apply
        
        Returns:
        --------
        self : FeatureEngineer
            Returns self for method chaining
        """
        for agg_col in agg_cols:
            for func in agg_funcs:
                feature_name = f"{agg_col}_{func}_by_{group_by_col}"
                agg_values = self.data.groupby(group_by_col)[agg_col].transform(func)
                self.data[feature_name] = agg_values
                self.feature_history.append(f"Created aggregate: {feature_name}")
        
        return self
    
    def create_binned_features(self, column, n_bins=5, strategy='quantile'):
        """
        Create binned categorical features from continuous variables.
        
        Parameters:
        -----------
        column : str
            Column to bin
        n_bins : int, default=5
            Number of bins to create
        strategy : str, default='quantile'
            Strategy for binning: 'quantile', 'uniform', 'kmeans'
        
        Returns:
        --------
        self : FeatureEngineer
            Returns self for method chaining
        """
        binned_col_name = f"{column}_binned"
        
        if strategy == 'quantile':
            self.data[binned_col_name] = pd.qcut(self.data[column], q=n_bins, labels=False, duplicates='drop')
        elif strategy == 'uniform':
            self.data[binned_col_name] = pd.cut(self.data[column], bins=n_bins, labels=False)
        
        self.feature_history.append(f"Created binned feature: {binned_col_name}")
        return self
    
    def encode_categorical_onehot(self, columns, drop_first=False):
        """
        One-hot encode categorical variables.
        
        Parameters:
        -----------
        columns : list
            Columns to one-hot encode
        drop_first : bool, default=False
            Whether to drop the first category
        
        Returns:
        --------
        self : FeatureEngineer
            Returns self for method chaining
        """
        for col in columns:
            dummies = pd.get_dummies(self.data[col], prefix=col, drop_first=drop_first)
            self.data = pd.concat([self.data, dummies], axis=1)
            self.data = self.data.drop(columns=[col])
            self.feature_history.append(f"One-hot encoded: {col}")
        
        return self
    
    def encode_categorical_target(self, columns, target_col):
        """
        Target encode categorical variables.
        
        Parameters:
        -----------
        columns : list
            Columns to target encode
        target_col : str
            Target column for encoding
        
        Returns:
        --------
        self : FeatureEngineer
            Returns self for method chaining
        """
        encoder = TargetEncoder()
        
        for col in columns:
            encoded_col_name = f"{col}_target_encoded"
            self.data[encoded_col_name] = encoder.fit_transform(
                self.data[col], 
                self.data[target_col]
            )
            self.feature_history.append(f"Target encoded: {col}")
        
        return self
    
    def create_datetime_features(self, date_column):
        """
        Extract features from datetime columns.
        
        Parameters:
        -----------
        date_column : str
            Name of the datetime column
        
        Returns:
        --------
        self : FeatureEngineer
            Returns self for method chaining
        """
        # Ensure column is datetime
        self.data[date_column] = pd.to_datetime(self.data[date_column])
        
        # Extract features
        self.data[f'{date_column}_year'] = self.data[date_column].dt.year
        self.data[f'{date_column}_month'] = self.data[date_column].dt.month
        self.data[f'{date_column}_day'] = self.data[date_column].dt.day
        self.data[f'{date_column}_dayofweek'] = self.data[date_column].dt.dayofweek
        self.data[f'{date_column}_quarter'] = self.data[date_column].dt.quarter
        self.data[f'{date_column}_is_weekend'] = self.data[date_column].dt.dayofweek.isin([5, 6]).astype(int)
        
        self.feature_history.append(f"Extracted datetime features from: {date_column}")
        return self
    
    def create_text_features(self, text_column):
        """
        Extract basic features from text columns.
        
        Parameters:
        -----------
        text_column : str
            Name of the text column
        
        Returns:
        --------
        self : FeatureEngineer
            Returns self for method chaining
        """
        self.data[f'{text_column}_length'] = self.data[text_column].str.len()
        self.data[f'{text_column}_word_count'] = self.data[text_column].str.split().str.len()
        self.data[f'{text_column}_unique_words'] = self.data[text_column].apply(
            lambda x: len(set(str(x).split()))
        )
        
        self.feature_history.append(f"Extracted text features from: {text_column}")
        return self
    
    def get_engineered_data(self):
        """
        Get the data with engineered features.
        
        Returns:
        --------
        pd.DataFrame : Data with engineered features
        """
        return self.data
    
    def get_feature_history(self):
        """
        Get the history of feature engineering operations.
        
        Returns:
        --------
        list : List of feature engineering operations
        """
        return self.feature_history


class FeatureSelector:
    """
    Feature selection methods.
    """
    
    @staticmethod
    def select_k_best(X, y, k=10, score_func=f_classif):
        """
        Select K best features using statistical tests.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        y : pd.Series or np.array
            Target variable
        k : int, default=10
            Number of features to select
        score_func : callable, default=f_classif
            Score function for feature evaluation
        
        Returns:
        --------
        tuple : Selected feature indices and scores
        """
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        return selector.get_support(indices=True), selector.scores_
    
    @staticmethod
    def select_by_importance(X, y, n_features=10):
        """
        Select features based on Random Forest importance.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        y : pd.Series or np.array
            Target variable
        n_features : int, default=10
            Number of top features to select
        
        Returns:
        --------
        tuple : Selected feature indices and importances
        """
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1][:n_features]
        
        return indices, importances[indices]
    
    @staticmethod
    def recursive_feature_elimination(X, y, n_features=10):
        """
        Perform Recursive Feature Elimination.
        
        Parameters:
        -----------
        X : pd.DataFrame or np.array
            Feature matrix
        y : pd.Series or np.array
            Target variable
        n_features : int, default=10
            Number of features to select
        
        Returns:
        --------
        np.array : Selected feature indices
        """
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe = RFE(estimator=estimator, n_features_to_select=n_features)
        rfe.fit(X, y)
        
        return np.where(rfe.support_)[0]


# Demonstrations

def demonstrate_feature_engineering():
    """
    Demonstrate feature engineering techniques.
    """
    print("=" * 80)
    print("FEATURE ENGINEERING DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'loan_amount': np.random.normal(20000, 10000, n_samples),
        'employment_length': np.random.randint(0, 30, n_samples),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'date': pd.date_range(start='2020-01-01', periods=n_samples, freq='D')
    })
    
    print(f"\n1. Original Data Shape: {df.shape}")
    print(f"\n2. Original Features: {list(df.columns)}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(df)
    
    # Create various features
    engineer = (engineer
                .create_polynomial_features(['age', 'income'], degree=2)
                .create_interactions([('income', 'credit_score'), ('age', 'employment_length')])
                .create_ratio_features(['loan_amount'], ['income'])
                .create_binned_features('age', n_bins=5)
                .create_datetime_features('date'))
    
    engineered_df = engineer.get_engineered_data()
    
    print(f"\n3. Engineered Data Shape: {engineered_df.shape}")
    print(f"\n4. New Features Created: {engineered_df.shape[1] - df.shape[1]}")
    
    print("\n5. Feature Engineering History:")
    for i, operation in enumerate(engineer.get_feature_history(), 1):
        print(f"   {i}. {operation}")
    
    print("\n6. Sample of New Features:")
    new_cols = [col for col in engineered_df.columns if col not in df.columns]
    print(f"   {new_cols[:10]}...")
    
    print("\n" + "=" * 80)
    
    return engineered_df


def demonstrate_feature_selection():
    """
    Demonstrate feature selection techniques.
    """
    print("\n" + "=" * 80)
    print("FEATURE SELECTION DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data with target
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randn(n_samples, n_features)
    # Make some features more relevant
    y = (X[:, 0] + X[:, 1] * 2 + X[:, 2] * 3 + np.random.randn(n_samples) * 0.1 > 0).astype(int)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    
    print(f"\n1. Dataset: {n_samples} samples, {n_features} features")
    
    # Method 1: SelectKBest
    print("\n2. SelectKBest (k=5):")
    k_best_indices, k_best_scores = FeatureSelector.select_k_best(X, y, k=5)
    print(f"   Selected features: {[feature_names[i] for i in k_best_indices]}")
    print(f"   Scores: {k_best_scores}")
    
    # Method 2: Feature Importance
    print("\n3. Random Forest Importance (top 5):")
    importance_indices, importances = FeatureSelector.select_by_importance(X, y, n_features=5)
    print(f"   Selected features: {[feature_names[i] for i in importance_indices]}")
    print(f"   Importances: {importances}")
    
    # Method 3: RFE
    print("\n4. Recursive Feature Elimination (n=5):")
    rfe_indices = FeatureSelector.recursive_feature_elimination(X, y, n_features=5)
    print(f"   Selected features: {[feature_names[i] for i in rfe_indices]}")
    
    print("\n" + "=" * 80)


def demonstrate_domain_specific_features():
    """
    Demonstrate domain-specific feature engineering for financial data.
    """
    print("\n" + "=" * 80)
    print("DOMAIN-SPECIFIC FEATURE ENGINEERING (Financial Data)")
    print("=" * 80)
    
    # Create sample financial transaction data
    np.random.seed(42)
    n_samples = 500
    
    df = pd.DataFrame({
        'transaction_amount': np.random.exponential(100, n_samples),
        'account_balance': np.random.normal(5000, 2000, n_samples),
        'transaction_count_30d': np.random.poisson(15, n_samples),
        'avg_transaction_30d': np.random.normal(80, 30, n_samples),
        'days_since_last_transaction': np.random.randint(0, 30, n_samples),
        'account_age_days': np.random.randint(30, 1000, n_samples)
    })
    
    print(f"\n1. Original Financial Data Shape: {df.shape}")
    
    # Create domain-specific features
    engineer = FeatureEngineer(df)
    
    # Financial ratios
    df_engineered = engineer.data
    df_engineered['transaction_to_balance_ratio'] = (
        df_engineered['transaction_amount'] / (df_engineered['account_balance'] + 1e-8)
    )
    df_engineered['transaction_velocity'] = (
        df_engineered['transaction_count_30d'] / (df_engineered['days_since_last_transaction'] + 1)
    )
    df_engineered['balance_per_day'] = (
        df_engineered['account_balance'] / (df_engineered['account_age_days'] + 1)
    )
    
    # Statistical features
    df_engineered['transaction_zscore'] = (
        (df_engineered['transaction_amount'] - df_engineered['avg_transaction_30d']) / 
        (df_engineered['avg_transaction_30d'].std() + 1e-8)
    )
    
    print(f"\n2. Engineered Financial Data Shape: {df_engineered.shape}")
    print(f"\n3. New Financial Features:")
    new_features = ['transaction_to_balance_ratio', 'transaction_velocity', 
                   'balance_per_day', 'transaction_zscore']
    print(f"   {new_features}")
    
    print(f"\n4. Sample Statistics:")
    print(df_engineered[new_features].describe())
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    engineered_data = demonstrate_feature_engineering()
    demonstrate_feature_selection()
    demonstrate_domain_specific_features()
    
    print("\n✅ Module 3 Complete: Feature Engineering")
    print("\nKey Takeaways:")
    print("1. Feature engineering can significantly improve model performance")
    print("2. Domain knowledge is crucial for creating meaningful features")
    print("3. Polynomial and interaction features capture non-linear relationships")
    print("4. Feature selection helps reduce dimensionality and overfitting")
    print("5. Always validate new features with cross-validation")
