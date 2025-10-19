"""
Day 1 - Module 2: Outlier Detection and Treatment
Topic: Identifying and Handling Outliers in Data

This module covers:
- Statistical methods for outlier detection (Z-score, IQR)
- Machine learning-based outlier detection
- Outlier treatment strategies
- Robust scaling techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import RobustScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class OutlierDetector:
    """
    A comprehensive class for detecting and handling outliers in datasets.
    """
    
    def __init__(self, data):
        """
        Initialize the OutlierDetector.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data for outlier detection
        """
        self.data = data.copy()
        self.outlier_indices = {}
        
    def zscore_method(self, columns=None, threshold=3):
        """
        Detect outliers using Z-score method.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to check for outliers. If None, uses all numeric columns.
        threshold : float, default=3
            Z-score threshold for outlier detection
        
        Returns:
        --------
        dict : Dictionary mapping column names to outlier indices
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        for col in columns:
            z_scores = np.abs(stats.zscore(self.data[col].dropna()))
            outlier_idx = np.where(z_scores > threshold)[0]
            outliers[col] = outlier_idx
            
        self.outlier_indices['zscore'] = outliers
        return outliers
    
    def iqr_method(self, columns=None, multiplier=1.5):
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to check for outliers. If None, uses all numeric columns.
        multiplier : float, default=1.5
            IQR multiplier for determining outlier boundaries
        
        Returns:
        --------
        dict : Dictionary mapping column names to outlier indices
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        outliers = {}
        for col in columns:
            Q1 = self.data[col].quantile(0.25)
            Q3 = self.data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
            outlier_idx = np.where(outlier_mask)[0]
            outliers[col] = outlier_idx
            
        self.outlier_indices['iqr'] = outliers
        return outliers
    
    def isolation_forest_method(self, columns=None, contamination=0.1):
        """
        Detect outliers using Isolation Forest algorithm.
        
        Parameters:
        -----------
        columns : list, optional
            Columns to use for detection. If None, uses all numeric columns.
        contamination : float, default=0.1
            Expected proportion of outliers in the dataset
        
        Returns:
        --------
        np.array : Boolean array indicating outliers
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = self.data[columns].dropna()
        
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        outlier_labels = iso_forest.fit_predict(X)
        
        # -1 indicates outlier, 1 indicates inlier
        outlier_mask = outlier_labels == -1
        
        self.outlier_indices['isolation_forest'] = np.where(outlier_mask)[0]
        return outlier_mask
    
    def lof_method(self, columns=None, n_neighbors=20, contamination=0.1):
        """
        Detect outliers using Local Outlier Factor (LOF).
        
        Parameters:
        -----------
        columns : list, optional
            Columns to use for detection. If None, uses all numeric columns.
        n_neighbors : int, default=20
            Number of neighbors to consider
        contamination : float, default=0.1
            Expected proportion of outliers
        
        Returns:
        --------
        np.array : Boolean array indicating outliers
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = self.data[columns].dropna()
        
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        outlier_labels = lof.fit_predict(X)
        
        outlier_mask = outlier_labels == -1
        
        self.outlier_indices['lof'] = np.where(outlier_mask)[0]
        return outlier_mask
    
    def elliptic_envelope_method(self, columns=None, contamination=0.1):
        """
        Detect outliers using Elliptic Envelope (assumes Gaussian distribution).
        
        Parameters:
        -----------
        columns : list, optional
            Columns to use for detection. If None, uses all numeric columns.
        contamination : float, default=0.1
            Expected proportion of outliers
        
        Returns:
        --------
        np.array : Boolean array indicating outliers
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        X = self.data[columns].dropna()
        
        envelope = EllipticEnvelope(contamination=contamination, random_state=42)
        outlier_labels = envelope.fit_predict(X)
        
        outlier_mask = outlier_labels == -1
        
        self.outlier_indices['elliptic_envelope'] = np.where(outlier_mask)[0]
        return outlier_mask
    
    def get_outlier_summary(self):
        """
        Get a summary of detected outliers across all methods.
        
        Returns:
        --------
        pd.DataFrame : Summary of outliers detected by each method
        """
        summary = {}
        for method, indices in self.outlier_indices.items():
            if isinstance(indices, dict):
                total_outliers = sum(len(idx) for idx in indices.values())
            else:
                total_outliers = len(indices)
            summary[method] = total_outliers
        
        return pd.DataFrame([summary]).T.rename(columns={0: 'outliers_detected'})


class OutlierTreatment:
    """
    Methods for treating/handling detected outliers.
    """
    
    @staticmethod
    def remove_outliers(data, outlier_indices):
        """
        Remove outlier rows from the dataset.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        outlier_indices : np.array or list
            Indices of outliers to remove
        
        Returns:
        --------
        pd.DataFrame : Data with outliers removed
        """
        return data.drop(index=outlier_indices).reset_index(drop=True)
    
    @staticmethod
    def cap_outliers(data, columns, lower_percentile=5, upper_percentile=95):
        """
        Cap outliers at specified percentiles (Winsorization).
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        columns : list
            Columns to cap
        lower_percentile : float, default=5
            Lower percentile for capping
        upper_percentile : float, default=95
            Upper percentile for capping
        
        Returns:
        --------
        pd.DataFrame : Data with capped values
        """
        data_capped = data.copy()
        
        for col in columns:
            lower_bound = data[col].quantile(lower_percentile / 100)
            upper_bound = data[col].quantile(upper_percentile / 100)
            
            data_capped[col] = data_capped[col].clip(lower=lower_bound, upper=upper_bound)
        
        return data_capped
    
    @staticmethod
    def transform_outliers(data, columns, method='log'):
        """
        Transform data to reduce impact of outliers.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        columns : list
            Columns to transform
        method : str, default='log'
            Transformation method: 'log', 'sqrt', 'boxcox'
        
        Returns:
        --------
        pd.DataFrame : Transformed data
        """
        data_transformed = data.copy()
        
        for col in columns:
            if method == 'log':
                # Add 1 to handle zero values
                data_transformed[col] = np.log1p(data[col])
            elif method == 'sqrt':
                data_transformed[col] = np.sqrt(data[col])
            elif method == 'boxcox':
                # Box-Cox requires positive values
                if (data[col] > 0).all():
                    data_transformed[col], _ = stats.boxcox(data[col])
                else:
                    print(f"Warning: {col} contains non-positive values, skipping Box-Cox")
        
        return data_transformed
    
    @staticmethod
    def robust_scaling(data, columns):
        """
        Apply robust scaling that is resistant to outliers.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data
        columns : list
            Columns to scale
        
        Returns:
        --------
        pd.DataFrame : Scaled data
        """
        data_scaled = data.copy()
        scaler = RobustScaler()
        
        data_scaled[columns] = scaler.fit_transform(data[columns])
        
        return data_scaled


# Demonstrations and Examples

def demonstrate_outlier_detection():
    """
    Demonstrate various outlier detection methods.
    """
    print("=" * 80)
    print("OUTLIER DETECTION DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data with outliers
    np.random.seed(42)
    n_samples = 500
    n_outliers = 25
    
    # Normal data
    normal_data = np.random.normal(100, 15, n_samples)
    
    # Add outliers
    outliers = np.random.uniform(200, 250, n_outliers)
    data_with_outliers = np.concatenate([normal_data, outliers])
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature1': data_with_outliers,
        'feature2': np.random.normal(50, 10, len(data_with_outliers)),
        'feature3': np.random.exponential(20, len(data_with_outliers))
    })
    
    print(f"\n1. Dataset Shape: {df.shape}")
    print(f"\n2. Basic Statistics:")
    print(df.describe())
    
    # Initialize detector
    detector = OutlierDetector(df)
    
    # Method 1: Z-score
    print("\n3. Z-score Method (threshold=3):")
    zscore_outliers = detector.zscore_method(threshold=3)
    for col, indices in zscore_outliers.items():
        print(f"   - {col}: {len(indices)} outliers detected")
    
    # Method 2: IQR
    print("\n4. IQR Method (multiplier=1.5):")
    iqr_outliers = detector.iqr_method(multiplier=1.5)
    for col, indices in iqr_outliers.items():
        print(f"   - {col}: {len(indices)} outliers detected")
    
    # Method 3: Isolation Forest
    print("\n5. Isolation Forest (contamination=0.1):")
    iso_outliers = detector.isolation_forest_method(contamination=0.1)
    print(f"   - Total outliers detected: {iso_outliers.sum()}")
    
    # Method 4: LOF
    print("\n6. Local Outlier Factor (contamination=0.1):")
    lof_outliers = detector.lof_method(contamination=0.1)
    print(f"   - Total outliers detected: {lof_outliers.sum()}")
    
    # Summary
    print("\n7. Outlier Detection Summary:")
    summary = detector.get_outlier_summary()
    print(summary)
    
    print("\n" + "=" * 80)
    
    return df, detector


def demonstrate_outlier_treatment():
    """
    Demonstrate various outlier treatment methods.
    """
    print("\n" + "=" * 80)
    print("OUTLIER TREATMENT DEMONSTRATION")
    print("=" * 80)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 300
    
    df = pd.DataFrame({
        'price': np.concatenate([
            np.random.normal(100, 15, n_samples),
            np.random.uniform(300, 500, 30)  # outliers
        ]),
        'quantity': np.concatenate([
            np.random.poisson(50, n_samples),
            np.random.uniform(200, 300, 30)  # outliers
        ])
    })
    
    print(f"\n1. Original Data Statistics:")
    print(df.describe())
    
    # Treatment 1: Remove outliers
    print("\n2. Treatment 1: Remove Outliers")
    detector = OutlierDetector(df)
    outliers = detector.isolation_forest_method(contamination=0.1)
    df_removed = OutlierTreatment.remove_outliers(df, detector.outlier_indices['isolation_forest'])
    print(f"   - Original size: {len(df)}")
    print(f"   - After removal: {len(df_removed)}")
    print(f"   - Removed: {len(df) - len(df_removed)} rows")
    
    # Treatment 2: Cap outliers
    print("\n3. Treatment 2: Cap Outliers (Winsorization)")
    df_capped = OutlierTreatment.cap_outliers(
        df, 
        columns=['price', 'quantity'],
        lower_percentile=5,
        upper_percentile=95
    )
    print(f"   - Original price range: [{df['price'].min():.2f}, {df['price'].max():.2f}]")
    print(f"   - Capped price range: [{df_capped['price'].min():.2f}, {df_capped['price'].max():.2f}]")
    
    # Treatment 3: Transform outliers
    print("\n4. Treatment 3: Log Transformation")
    df_transformed = OutlierTreatment.transform_outliers(
        df,
        columns=['price', 'quantity'],
        method='log'
    )
    print(f"   - Original price std: {df['price'].std():.2f}")
    print(f"   - Transformed price std: {df_transformed['price'].std():.2f}")
    
    # Treatment 4: Robust scaling
    print("\n5. Treatment 4: Robust Scaling")
    df_scaled = OutlierTreatment.robust_scaling(df, columns=['price', 'quantity'])
    print(f"   - Scaled price range: [{df_scaled['price'].min():.2f}, {df_scaled['price'].max():.2f}]")
    print(f"   - Scaled price median: {df_scaled['price'].median():.2f}")
    
    print("\n" + "=" * 80)
    
    return df, df_capped, df_transformed, df_scaled


if __name__ == "__main__":
    # Run demonstrations
    df_original, detector = demonstrate_outlier_detection()
    df_variants = demonstrate_outlier_treatment()
    
    print("\n✅ Module 2 Complete: Outlier Detection and Treatment")
    print("\nKey Takeaways:")
    print("1. Different methods detect different types of outliers")
    print("2. Statistical methods (Z-score, IQR) work well for univariate analysis")
    print("3. ML methods (Isolation Forest, LOF) detect multivariate outliers")
    print("4. Choose treatment based on domain knowledge and model requirements")
    print("5. Sometimes outliers are valuable data points, not errors!")
