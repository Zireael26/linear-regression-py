"""
Dataset Generation Script for Linear Regression Testing

This script generates various datasets for testing linear regression implementations:
- Simple linear relationships
- Multivariate datasets
- Datasets with different noise levels
- Datasets with outliers
- Real-world inspired datasets
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)


def generate_simple_linear_data(n_samples=1000, noise=0.1, slope=2.5, intercept=1.0):
    """
    Generate simple 1D linear data: y = slope * x + intercept + noise
    
    Args:
        n_samples (int): Number of samples
        noise (float): Standard deviation of noise
        slope (float): True slope
        intercept (float): True intercept
        
    Returns:
        tuple: (X, y) arrays
    """
    X = np.random.uniform(-10, 10, n_samples)
    y = slope * X + intercept + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y


def generate_multivariate_data(n_samples=1000, n_features=5, noise=0.2):
    """
    Generate multivariate linear data with known coefficients.
    
    Args:
        n_samples (int): Number of samples
        n_features (int): Number of features
        noise (float): Standard deviation of noise
        
    Returns:
        tuple: (X, y, true_coefficients) arrays
    """
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Define true coefficients
    true_coefficients = np.random.uniform(-3, 3, n_features)
    true_intercept = np.random.uniform(-2, 2)
    
    # Generate target variable
    y = X @ true_coefficients + true_intercept + np.random.normal(0, noise, n_samples)
    
    return X, y, true_coefficients, true_intercept


def generate_housing_dataset(n_samples=2000):
    """
    Generate a realistic housing price dataset.
    
    Features:
    - House size (sq ft)
    - Number of bedrooms
    - Number of bathrooms
    - Age of house
    - Distance to city center
    
    Returns:
        tuple: (X, y, feature_names)
    """
    # Generate correlated features
    house_size = np.random.normal(2000, 600, n_samples)
    house_size = np.clip(house_size, 800, 5000)
    
    # Bedrooms correlated with house size
    bedrooms = np.round(house_size / 500 + np.random.normal(0, 0.5, n_samples))
    bedrooms = np.clip(bedrooms, 1, 6)
    
    # Bathrooms correlated with bedrooms
    bathrooms = bedrooms * 0.75 + np.random.normal(0, 0.3, n_samples)
    bathrooms = np.clip(bathrooms, 1, 4)
    
    # House age
    age = np.random.exponential(15, n_samples)
    age = np.clip(age, 0, 100)
    
    # Distance to city center
    distance = np.random.gamma(2, 5, n_samples)
    distance = np.clip(distance, 1, 50)
    
    # Combine features
    X = np.column_stack([house_size, bedrooms, bathrooms, age, distance])
    
    # Generate price with realistic coefficients
    price = (
        house_size * 150 +          # $150 per sq ft
        bedrooms * 10000 +          # $10k per bedroom
        bathrooms * 15000 +         # $15k per bathroom
        age * (-800) +              # Depreciation
        distance * (-2000) +        # Distance penalty
        200000 +                    # Base price
        np.random.normal(0, 25000, n_samples)  # Noise
    )
    
    price = np.clip(price, 50000, 2000000)  # Realistic price range
    
    feature_names = ['house_size_sqft', 'bedrooms', 'bathrooms', 'age_years', 'distance_miles']
    
    return X, price, feature_names


def generate_polynomial_data(n_samples=800, degree=2, noise=0.3):
    """
    Generate data with polynomial relationship (for testing overfitting).
    
    Args:
        n_samples (int): Number of samples
        degree (int): Degree of polynomial
        noise (float): Standard deviation of noise
        
    Returns:
        tuple: (X, y) arrays
    """
    X = np.random.uniform(-3, 3, n_samples)
    
    # Generate polynomial features
    y = 0
    for i in range(degree + 1):
        coef = np.random.uniform(-1, 1)
        y += coef * (X ** i)
    
    y += np.random.normal(0, noise, n_samples)
    
    return X.reshape(-1, 1), y


def generate_data_with_outliers(n_samples=500, outlier_fraction=0.05):
    """
    Generate linear data with outliers.
    
    Args:
        n_samples (int): Number of samples
        outlier_fraction (float): Fraction of samples that are outliers
        
    Returns:
        tuple: (X, y) arrays
    """
    # Generate normal data
    X = np.random.uniform(-5, 5, n_samples)
    y = 2.0 * X + 1.0 + np.random.normal(0, 0.5, n_samples)
    
    # Add outliers
    n_outliers = int(n_samples * outlier_fraction)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    
    # Make outliers have large y values
    y[outlier_indices] += np.random.uniform(10, 20, n_outliers) * np.random.choice([-1, 1], n_outliers)
    
    return X.reshape(-1, 1), y


def generate_heteroscedastic_data(n_samples=800):
    """
    Generate data with non-constant variance (heteroscedasticity).
    
    Returns:
        tuple: (X, y) arrays
    """
    X = np.random.uniform(0, 10, n_samples)
    
    # Variance increases with X
    noise_std = 0.1 + 0.3 * X
    noise = np.random.normal(0, noise_std)
    
    y = 2.0 * X + 1.0 + noise
    
    return X.reshape(-1, 1), y


def generate_time_series_data(n_samples=1000):
    """
    Generate time series data with trend.
    
    Returns:
        tuple: (time, values) arrays
    """
    time = np.arange(n_samples)
    
    # Linear trend with seasonal component
    trend = 0.1 * time
    seasonal = 2 * np.sin(2 * np.pi * time / 50)  # 50-day cycle
    noise = np.random.normal(0, 0.5, n_samples)
    
    values = trend + seasonal + noise + 10  # Base level
    
    return time.reshape(-1, 1), values


def save_dataset(X, y, filename, feature_names=None, additional_info=None):
    """Save dataset to CSV file."""
    data_dir = Path(__file__).parent
    
    if X.shape[1] == 1:
        df = pd.DataFrame({'X': X.ravel(), 'y': y})
    else:
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
    
    # Add metadata as comments in the first few rows
    filepath = data_dir / filename
    with open(filepath, 'w') as f:
        f.write(f"# Dataset: {filename}\n")
        f.write(f"# Samples: {len(y)}\n")
        f.write(f"# Features: {X.shape[1]}\n")
        if additional_info:
            for key, value in additional_info.items():
                f.write(f"# {key}: {value}\n")
        f.write("#\n")
    
    # Append the actual data
    df.to_csv(filepath, mode='a', index=False)
    print(f"Saved {filename} with {len(y)} samples and {X.shape[1]} features")


def create_visualization(datasets_info):
    """Create visualization of generated datasets."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, (name, X, y) in enumerate(datasets_info[:6]):
        ax = axes[i]
        
        if X.shape[1] == 1:
            ax.scatter(X.ravel(), y, alpha=0.6, s=1)
            ax.set_xlabel('X')
        else:
            # For multivariate, plot first feature vs target
            ax.scatter(X[:, 0], y, alpha=0.6, s=1)
            ax.set_xlabel('First Feature')
        
        ax.set_ylabel('Target')
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'dataset_overview.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Generate all datasets."""
    print("Generating datasets for linear regression testing...")
    set_random_seed(42)
    
    datasets_info = []
    
    # 1. Simple linear data (different noise levels)
    for noise_level, suffix in [(0.1, 'low_noise'), (0.5, 'medium_noise'), (1.0, 'high_noise')]:
        X, y = generate_simple_linear_data(n_samples=1000, noise=noise_level)
        save_dataset(X, y, f'simple_linear_{suffix}.csv', 
                    additional_info={'noise_std': noise_level, 'true_slope': 2.5, 'true_intercept': 1.0})
        datasets_info.append((f'Simple Linear ({suffix})', X, y))
    
    # 2. Multivariate data
    X, y, true_coef, true_int = generate_multivariate_data(n_samples=1500, n_features=5)
    save_dataset(X, y, 'multivariate_5features.csv',
                additional_info={'true_coefficients': true_coef.tolist(), 'true_intercept': true_int})
    datasets_info.append(('Multivariate (5 features)', X, y))
    
    # 3. Housing dataset
    X, y, feature_names = generate_housing_dataset(n_samples=2000)
    save_dataset(X, y, 'housing_prices.csv', feature_names=feature_names,
                additional_info={'description': 'Synthetic housing price dataset'})
    datasets_info.append(('Housing Prices', X, y))
    
    # 4. Large multivariate dataset
    X, y, true_coef, true_int = generate_multivariate_data(n_samples=5000, n_features=10, noise=0.3)
    save_dataset(X, y, 'large_multivariate.csv',
                additional_info={'description': 'Large dataset for performance testing'})
    datasets_info.append(('Large Multivariate', X, y))
    
    # 5. Data with outliers
    X, y = generate_data_with_outliers(n_samples=800, outlier_fraction=0.08)
    save_dataset(X, y, 'data_with_outliers.csv',
                additional_info={'outlier_fraction': 0.08, 'description': 'Linear data with outliers'})
    datasets_info.append(('Data with Outliers', X, y))
    
    # 6. Heteroscedastic data
    X, y = generate_heteroscedastic_data(n_samples=1000)
    save_dataset(X, y, 'heteroscedastic_data.csv',
                additional_info={'description': 'Data with non-constant variance'})
    datasets_info.append(('Heteroscedastic Data', X, y))
    
    # 7. Time series data
    X, y = generate_time_series_data(n_samples=1000)
    save_dataset(X, y, 'time_series_trend.csv',
                additional_info={'description': 'Time series with linear trend and seasonality'})
    datasets_info.append(('Time Series', X, y))
    
    # 8. Polynomial data (for testing overfitting)
    X, y = generate_polynomial_data(n_samples=600, degree=3, noise=0.4)
    save_dataset(X, y, 'polynomial_degree3.csv',
                additional_info={'degree': 3, 'description': 'Polynomial relationship (degree 3)'})
    
    # 9. Very large dataset for performance testing
    X, y = generate_simple_linear_data(n_samples=50000, noise=0.2)
    save_dataset(X, y, 'large_simple_linear.csv',
                additional_info={'description': 'Large simple linear dataset for performance testing'})
    
    # 10. Small dataset for edge case testing
    X, y = generate_simple_linear_data(n_samples=20, noise=0.1)
    save_dataset(X, y, 'small_dataset.csv',
                additional_info={'description': 'Small dataset for edge case testing'})
    
    print(f"\nGenerated {len(datasets_info) + 3} datasets successfully!")
    print("\nDataset Summary:")
    print("- Simple linear (3 noise levels): 1000 samples each")
    print("- Multivariate (5 features): 1500 samples")
    print("- Housing prices: 2000 samples, 5 features")
    print("- Large multivariate (10 features): 5000 samples")
    print("- Data with outliers: 800 samples")
    print("- Heteroscedastic data: 1000 samples")
    print("- Time series: 1000 samples")
    print("- Polynomial (degree 3): 600 samples")
    print("- Large simple linear: 50,000 samples")
    print("- Small dataset: 20 samples")
    
    # Create visualization
    create_visualization(datasets_info)


if __name__ == "__main__":
    main()
