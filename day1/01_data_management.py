"""
Day 1 - Module 1: Data Management
Topic: Handling Dirty Data and Missing Values

This module covers techniques for managing real-world messy data including:
- Identifying and handling missing values
- Data validation and cleaning
- Data type conversions
- Duplicate detection and removal
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class DataCleaner:
    """
    A comprehensive data cleaning class for handling common data quality issues.
    """
    
    def __init__(self, data):
        """
        Initialize the DataCleaner with a DataFrame.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data to be cleaned
        """
        self.data = data.copy()
        self.original_shape = data.shape
        self.cleaning_report = []
    
    def identify_missing(self):
        """
        Identify missing values in the dataset.
        
        Returns:
        --------
        pd.DataFrame : Summary of missing values per column
        """
        missing_summary = pd.DataFrame({
            'column': self.data.columns,
            'missing_count': self.data.isnull().sum().values,
            'missing_percentage': (self.data.isnull().sum().values / len(self.data) * 100)
        })
        missing_summary = missing_summary[missing_summary['missing_count'] > 0]
        missing_summary = missing_summary.sort_values('missing_percentage', ascending=False)
        
        self.cleaning_report.append(f"Missing values identified in {len(missing_summary)} columns")
        return missing_summary
    
    def handle_missing_numeric(self, strategy='mean', columns=None):
        """
        Handle missing values in numeric columns.
        
        Parameters:
        -----------
        strategy : str, default='mean'
            Strategy for imputation: 'mean', 'median', 'most_frequent', 'constant', 'knn'
        columns : list, optional
            Specific columns to apply imputation. If None, applies to all numeric columns.
        
        Returns:
        --------
        self : DataCleaner
            Returns self for method chaining
        """
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns:
            numeric_cols = [col for col in columns if col in numeric_cols]
        
        if strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=strategy)
        
        if numeric_cols:
            self.data[numeric_cols] = imputer.fit_transform(self.data[numeric_cols])
            self.cleaning_report.append(f"Imputed missing values in {len(numeric_cols)} numeric columns using {strategy} strategy")
        
        return self
    
    def handle_missing_categorical(self, strategy='most_frequent', columns=None):
        """
        Handle missing values in categorical columns.
        
        Parameters:
        -----------
        strategy : str, default='most_frequent'
            Strategy for imputation: 'most_frequent', 'constant'
        columns : list, optional
            Specific columns to apply imputation. If None, applies to all categorical columns.
        
        Returns:
        --------
        self : DataCleaner
            Returns self for method chaining
        """
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if columns:
            categorical_cols = [col for col in columns if col in categorical_cols]
        
        imputer = SimpleImputer(strategy=strategy, fill_value='Unknown')
        
        if categorical_cols:
            self.data[categorical_cols] = imputer.fit_transform(self.data[categorical_cols])
            self.cleaning_report.append(f"Imputed missing values in {len(categorical_cols)} categorical columns")
        
        return self
    
    def remove_duplicates(self, subset=None, keep='first'):
        """
        Remove duplicate rows from the dataset.
        
        Parameters:
        -----------
        subset : list, optional
            Column labels to consider for identifying duplicates
        keep : {'first', 'last', False}, default='first'
            Which duplicates to keep
        
        Returns:
        --------
        self : DataCleaner
            Returns self for method chaining
        """
        before_count = len(self.data)
        self.data = self.data.drop_duplicates(subset=subset, keep=keep)
        removed_count = before_count - len(self.data)
        
        if removed_count > 0:
            self.cleaning_report.append(f"Removed {removed_count} duplicate rows")
        
        return self
    
    def fix_data_types(self, type_mapping):
        """
        Convert columns to specified data types.
        
        Parameters:
        -----------
        type_mapping : dict
            Dictionary mapping column names to target data types
        
        Returns:
        --------
        self : DataCleaner
            Returns self for method chaining
        """
        for col, dtype in type_mapping.items():
            if col in self.data.columns:
                try:
                    self.data[col] = self.data[col].astype(dtype)
                    self.cleaning_report.append(f"Converted {col} to {dtype}")
                except Exception as e:
                    self.cleaning_report.append(f"Failed to convert {col} to {dtype}: {str(e)}")
        
        return self
    
    def remove_constant_columns(self):
        """
        Remove columns with constant values (no variance).
        
        Returns:
        --------
        self : DataCleaner
            Returns self for method chaining
        """
        constant_cols = [col for col in self.data.columns if self.data[col].nunique() == 1]
        
        if constant_cols:
            self.data = self.data.drop(columns=constant_cols)
            self.cleaning_report.append(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
        
        return self
    
    def get_clean_data(self):
        """
        Get the cleaned dataset.
        
        Returns:
        --------
        pd.DataFrame : Cleaned data
        """
        return self.data
    
    def get_cleaning_report(self):
        """
        Get a report of all cleaning operations performed.
        
        Returns:
        --------
        list : List of cleaning operations
        """
        return self.cleaning_report


# Example Usage and Demonstrations

def demonstrate_data_cleaning():
    """
    Demonstrate data cleaning techniques with a sample dataset.
    """
    print("=" * 80)
    print("DATA MANAGEMENT DEMONSTRATION")
    print("=" * 80)
    
    # Create a sample dirty dataset
    np.random.seed(42)
    n_samples = 1000
    
    dirty_data = pd.DataFrame({
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', None], n_samples),
        'employment': np.random.choice(['Employed', 'Unemployed', 'Self-Employed'], n_samples),
        'constant_col': 'same_value'  # Constant column
    })
    
    # Introduce missing values
    missing_mask = np.random.random((n_samples, 3)) < 0.1
    dirty_data.loc[missing_mask[:, 0], 'age'] = np.nan
    dirty_data.loc[missing_mask[:, 1], 'income'] = np.nan
    dirty_data.loc[missing_mask[:, 2], 'credit_score'] = np.nan
    
    # Introduce duplicates
    dirty_data = pd.concat([dirty_data, dirty_data.iloc[:50]], ignore_index=True)
    
    print(f"\n1. Original Data Shape: {dirty_data.shape}")
    print(f"\n2. Data Types:\n{dirty_data.dtypes}")
    
    # Initialize cleaner
    cleaner = DataCleaner(dirty_data)
    
    # Identify missing values
    print("\n3. Missing Values Summary:")
    missing_summary = cleaner.identify_missing()
    print(missing_summary)
    
    # Perform cleaning operations
    cleaner = (cleaner
               .remove_duplicates()
               .handle_missing_numeric(strategy='median')
               .handle_missing_categorical(strategy='most_frequent')
               .remove_constant_columns())
    
    # Get cleaned data
    clean_data = cleaner.get_clean_data()
    
    print(f"\n4. Cleaned Data Shape: {clean_data.shape}")
    print(f"\n5. Missing Values After Cleaning:\n{clean_data.isnull().sum()}")
    
    print("\n6. Cleaning Report:")
    for i, operation in enumerate(cleaner.get_cleaning_report(), 1):
        print(f"   {i}. {operation}")
    
    print("\n" + "=" * 80)
    print("Data cleaning completed successfully!")
    print("=" * 80)
    
    return clean_data


def advanced_missing_value_techniques():
    """
    Demonstrate advanced techniques for handling missing values.
    """
    print("\n" + "=" * 80)
    print("ADVANCED MISSING VALUE TECHNIQUES")
    print("=" * 80)
    
    # Create sample data with MCAR, MAR, and MNAR patterns
    np.random.seed(42)
    n = 500
    
    data = pd.DataFrame({
        'feature1': np.random.normal(100, 15, n),
        'feature2': np.random.normal(50, 10, n),
        'feature3': np.random.exponential(20, n),
        'target': np.random.choice([0, 1], n)
    })
    
    # MCAR: Missing Completely At Random
    mcar_mask = np.random.random(n) < 0.1
    data.loc[mcar_mask, 'feature1'] = np.nan
    
    # MAR: Missing At Random (dependent on another variable)
    mar_mask = data['feature2'] < 45
    data.loc[mar_mask & (np.random.random(n) < 0.2), 'feature3'] = np.nan
    
    print(f"\n1. Data with Missing Values:")
    print(f"   - Feature1 missing: {data['feature1'].isnull().sum()} (MCAR)")
    print(f"   - Feature3 missing: {data['feature3'].isnull().sum()} (MAR)")
    
    # Strategy 1: Mean Imputation
    print("\n2. Mean Imputation:")
    mean_imputer = SimpleImputer(strategy='mean')
    data_mean = data.copy()
    data_mean[['feature1', 'feature2', 'feature3']] = mean_imputer.fit_transform(
        data_mean[['feature1', 'feature2', 'feature3']]
    )
    print(f"   - Missing values after imputation: {data_mean.isnull().sum().sum()}")
    
    # Strategy 2: KNN Imputation
    print("\n3. KNN Imputation:")
    knn_imputer = KNNImputer(n_neighbors=5)
    data_knn = data.copy()
    data_knn[['feature1', 'feature2', 'feature3']] = knn_imputer.fit_transform(
        data_knn[['feature1', 'feature2', 'feature3']]
    )
    print(f"   - Missing values after imputation: {data_knn.isnull().sum().sum()}")
    
    # Strategy 3: Indicator Method (keeping track of missingness)
    print("\n4. Missing Value Indicator Method:")
    data_indicator = data.copy()
    data_indicator['feature1_missing'] = data['feature1'].isnull().astype(int)
    data_indicator['feature3_missing'] = data['feature3'].isnull().astype(int)
    data_indicator = data_indicator.fillna(data_indicator.mean())
    print(f"   - Added indicator columns for missingness")
    print(f"   - Feature1 was missing in {data_indicator['feature1_missing'].sum()} cases")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run demonstrations
    clean_data = demonstrate_data_cleaning()
    advanced_missing_value_techniques()
    
    print("\n✅ Module 1 Complete: Data Management")
    print("\nKey Takeaways:")
    print("1. Always identify missing value patterns before imputation")
    print("2. Different imputation strategies work better for different scenarios")
    print("3. Consider creating missing value indicators")
    print("4. Remove duplicates and constant columns early in the pipeline")
    print("5. Document all cleaning operations for reproducibility")
