import pandas as pd
import numpy as np

def generate_multicollinear_dataset(n_obs=100, n_features=3, n_noise_features = 3, feature_noise_level=1, noise_level=10):
    """
    Generates a dataset and target with collinear input features.
    
    The input dataset consists of two parts: 1) collinear signal features with added noise, and 2) completely random noise features.
    The target is the sum of the noise-free signals.
    
    Keyword arguments:
    n_obs: The number of observations to create.
    n_features: The number of signal features to create (x1, x2, and so on).
    n_noise_features: The number of pure noise features to create (noise1, noise2, and so on).
    feature_noise_level: The amplitude of the random noise added to the feature.
    noise_level: The amplitude of the pure noise features.
    """
    features = {}
    np.random.seed(1)
    for i in range(n_features):
        features[f'x{i}'] = np.arange(0,n_obs) + feature_noise_level * (np.random.rand(n_obs) - 0.5)
    for i in range(n_noise_features):
        features[f'noise{i}'] = noise_level*(np.random.rand(n_obs) - 0.5)
    X = pd.DataFrame(features)
    y = pd.Series(n_features * np.arange(0,100))
    return X, y
