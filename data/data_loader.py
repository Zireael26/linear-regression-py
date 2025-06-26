"""
Data Loading Utilities for Linear Regression Testing

This module provides convenient functions to load and work with
the generated datasets for testing linear regression implementations.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional


def get_data_directory() -> Path:
    """Get the path to the data directory."""
    return Path(__file__).parent


def list_available_datasets() -> List[str]:
    """
    List all available CSV datasets.
    
    Returns:
        List[str]: List of dataset filenames
    """
    data_dir = get_data_directory()
    datasets = [f.name for f in data_dir.glob("*.csv")]
    return sorted(datasets)


def load_dataset(filename: str, return_feature_names: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a dataset from CSV file.
    
    Args:
        filename (str): Name of the CSV file
        return_feature_names (bool): Whether to return feature names
        
    Returns:
        tuple: (X, y) arrays, and optionally feature names
    """
    data_dir = get_data_directory()
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset {filename} not found in {data_dir}")
    
    # Read CSV, skipping comment lines
    df = pd.read_csv(filepath, comment='#')
    
    # Determine features and target
    if 'target' in df.columns:
        # Multivariate case
        feature_cols = [col for col in df.columns if col != 'target']
        X = df[feature_cols].values
        y = df['target'].values
        feature_names = feature_cols
    elif 'y' in df.columns:
        # Simple case with X, y columns
        X = df['X'].values.reshape(-1, 1)
        y = df['y'].values
        feature_names = ['X']
    else:
        raise ValueError(f"Cannot determine target variable in {filename}")
    
    if return_feature_names:
        return X, y, feature_names
    return X, y


def get_dataset_info(filename: str) -> Dict[str, str]:
    """
    Extract metadata from dataset file.
    
    Args:
        filename (str): Name of the CSV file
        
    Returns:
        Dict[str, str]: Metadata dictionary
    """
    data_dir = get_data_directory()
    filepath = data_dir / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset {filename} not found in {data_dir}")
    
    info = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if ':' in line:
                    key, value = line[1:].strip().split(':', 1)
                    info[key.strip()] = value.strip()
            else:
                break
    
    return info


def print_dataset_summary(filename: str) -> None:
    """
    Print a summary of the dataset.
    
    Args:
        filename (str): Name of the CSV file
    """
    try:
        X, y, feature_names = load_dataset(filename, return_feature_names=True)
        info = get_dataset_info(filename)
        
        print(f"\n📊 Dataset: {filename}")
        print("=" * 50)
        print(f"Shape: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Features: {feature_names}")
        print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"Target mean: {y.mean():.3f} ± {y.std():.3f}")
        
        if info:
            print("\nMetadata:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        # Feature statistics for multivariate data
        if X.shape[1] > 1:
            print(f"\nFeature Statistics:")
            for i, name in enumerate(feature_names):
                print(f"  {name}: [{X[:, i].min():.3f}, {X[:, i].max():.3f}] "
                      f"(mean: {X[:, i].mean():.3f})")
    
    except Exception as e:
        print(f"Error loading {filename}: {e}")


def load_simple_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all simple (1D) datasets.
    
    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: Dictionary of dataset name to (X, y)
    """
    simple_datasets = {}
    
    simple_files = [
        'simple_linear_low_noise.csv',
        'simple_linear_medium_noise.csv', 
        'simple_linear_high_noise.csv',
        'data_with_outliers.csv',
        'heteroscedastic_data.csv',
        'time_series_trend.csv',
        'polynomial_degree3.csv',
        'small_dataset.csv'
    ]
    
    for filename in simple_files:
        try:
            X, y = load_dataset(filename)
            dataset_name = filename.replace('.csv', '').replace('_', ' ').title()
            simple_datasets[dataset_name] = (X, y)
        except FileNotFoundError:
            continue
    
    return simple_datasets


def load_multivariate_datasets() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load all multivariate datasets.
    
    Returns:
        Dict[str, Tuple[np.ndarray, np.ndarray]]: Dictionary of dataset name to (X, y)
    """
    multivariate_datasets = {}
    
    multivariate_files = [
        'multivariate_5features.csv',
        'housing_prices.csv',
        'large_multivariate.csv'
    ]
    
    for filename in multivariate_files:
        try:
            X, y = load_dataset(filename)
            dataset_name = filename.replace('.csv', '').replace('_', ' ').title()
            multivariate_datasets[dataset_name] = (X, y)
        except FileNotFoundError:
            continue
    
    return multivariate_datasets


def get_benchmark_dataset(size: str = 'medium') -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a dataset appropriate for benchmarking.
    
    Args:
        size (str): 'small', 'medium', or 'large'
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X, y) arrays
    """
    if size == 'small':
        return load_dataset('simple_linear_low_noise.csv')
    elif size == 'medium':
        return load_dataset('multivariate_5features.csv')
    elif size == 'large':
        return load_dataset('large_simple_linear.csv')
    else:
        raise ValueError("Size must be 'small', 'medium', or 'large'")


def split_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
               random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Target
        test_size (float): Fraction of data to use for testing
        random_state (int): Random seed
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    np.random.seed(random_state)
    n_samples = len(y)
    n_test = int(n_samples * test_size)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def main():
    """Display information about all available datasets."""
    print("📁 Available Datasets for Linear Regression Testing")
    print("=" * 60)
    
    datasets = list_available_datasets()
    if not datasets:
        print("No datasets found. Run generate_datasets.py first.")
        return
    
    for filename in datasets:
        print_dataset_summary(filename)
    
    print(f"\n🎯 Total datasets available: {len(datasets)}")
    print("\nQuick loading examples:")
    print("```python")
    print("from data.data_loader import load_dataset, split_data")
    print("")
    print("# Load a simple dataset")
    print("X, y = load_dataset('simple_linear_low_noise.csv')")
    print("")
    print("# Load with feature names")
    print("X, y, names = load_dataset('housing_prices.csv', return_feature_names=True)")
    print("")
    print("# Split into train/test")
    print("X_train, X_test, y_train, y_test = split_data(X, y)")
    print("```")


if __name__ == "__main__":
    main()
