"""
Dataset Examples and Usage Guide

This script demonstrates how to use the generated datasets
for testing your linear regression implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from data.data_loader import (
    load_dataset, 
    split_data, 
    list_available_datasets,
    get_benchmark_dataset,
    load_simple_datasets,
    load_multivariate_datasets
)


def demonstrate_simple_dataset():
    """Demonstrate loading and visualizing a simple dataset."""
    print("📊 Simple Dataset Example")
    print("-" * 30)
    
    # Load a simple linear dataset
    X, y = load_dataset('simple_linear_low_noise.csv')
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
    print(f"y range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Split into train/test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
    print(f"Training samples: {len(y_train)}")
    print(f"Testing samples: {len(y_test)}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.6, label='Training data', s=20)
    plt.scatter(X_test, y_test, alpha=0.8, label='Test data', s=20)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Simple Linear Dataset (Low Noise)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('simple_dataset_example.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_multivariate_dataset():
    """Demonstrate loading and exploring a multivariate dataset."""
    print("\n📊 Multivariate Dataset Example")
    print("-" * 35)
    
    # Load housing dataset with feature names
    X, y, feature_names = load_dataset('housing_prices.csv', return_feature_names=True)
    
    print(f"Dataset shape: X={X.shape}, y={y.shape}")
    print(f"Features: {feature_names}")
    print(f"Target (price) range: [${y.min():,.0f}, ${y.max():,.0f}]")
    
    # Show feature correlations with target
    print("\nFeature correlations with price:")
    for i, name in enumerate(feature_names):
        correlation = np.corrcoef(X[:, i], y)[0, 1]
        print(f"  {name}: {correlation:.3f}")
    
    # Visualize features vs target
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, name in enumerate(feature_names):
        axes[i].scatter(X[:, i], y, alpha=0.5, s=1)
        axes[i].set_xlabel(name.replace('_', ' ').title())
        axes[i].set_ylabel('Price ($)')
        axes[i].grid(True, alpha=0.3)
    
    # Distribution of target variable
    axes[5].hist(y, bins=50, alpha=0.7, edgecolor='black')
    axes[5].set_xlabel('Price ($)')
    axes[5].set_ylabel('Frequency')
    axes[5].set_title('Price Distribution')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('housing_dataset_example.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_outlier_dataset():
    """Demonstrate dataset with outliers."""
    print("\n📊 Dataset with Outliers Example")
    print("-" * 35)
    
    X, y = load_dataset('data_with_outliers.csv')
    
    # Identify potential outliers using z-score
    z_scores = np.abs((y - np.mean(y)) / np.std(y))
    outlier_threshold = 2.5
    outliers = z_scores > outlier_threshold
    
    print(f"Total samples: {len(y)}")
    print(f"Potential outliers (|z| > {outlier_threshold}): {np.sum(outliers)}")
    print(f"Outlier percentage: {100 * np.sum(outliers) / len(y):.1f}%")
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X[~outliers], y[~outliers], alpha=0.6, label='Normal data', s=20)
    plt.scatter(X[outliers], y[outliers], alpha=0.8, label='Potential outliers', 
                color='red', s=30)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Data with Outliers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(y, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('y values')
    plt.ylabel('Frequency')
    plt.title('Distribution showing outliers')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outlier_dataset_example.png', dpi=300, bbox_inches='tight')
    plt.show()


def demonstrate_noise_comparison():
    """Compare datasets with different noise levels."""
    print("\n📊 Noise Level Comparison")
    print("-" * 30)
    
    noise_levels = ['low_noise', 'medium_noise', 'high_noise']
    datasets = {}
    
    for noise in noise_levels:
        filename = f'simple_linear_{noise}.csv'
        X, y = load_dataset(filename)
        datasets[noise] = (X, y)
        print(f"{noise.replace('_', ' ').title()}: std = {np.std(y):.3f}")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, (noise, (X, y)) in enumerate(datasets.items()):
        axes[i].scatter(X, y, alpha=0.6, s=10)
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('y')
        axes[i].set_title(f'{noise.replace("_", " ").title()}')
        axes[i].grid(True, alpha=0.3)
        
        # Add true line
        x_line = np.linspace(X.min(), X.max(), 100)
        y_line = 2.5 * x_line + 1.0  # True relationship
        axes[i].plot(x_line, y_line, 'r--', alpha=0.8, label='True line')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('noise_comparison_example.png', dpi=300, bbox_inches='tight')
    plt.show()


def performance_testing_datasets():
    """Show datasets suitable for performance testing."""
    print("\n⚡ Performance Testing Datasets")
    print("-" * 35)
    
    performance_datasets = [
        ('small_dataset.csv', 'Edge case testing'),
        ('simple_linear_low_noise.csv', 'Standard testing'),
        ('large_multivariate.csv', 'Large multivariate'),
        ('large_simple_linear.csv', 'Large simple (50k samples)')
    ]
    
    for filename, description in performance_datasets:
        try:
            X, y = load_dataset(filename)
            print(f"{description:.<25} {X.shape[0]:>6} samples, {X.shape[1]:>2} features")
        except FileNotFoundError:
            print(f"{description:.<25} {'Not found':>12}")


def create_usage_examples():
    """Create example code snippets for using the datasets."""
    print("\n💻 Usage Examples")
    print("-" * 20)
    
    examples = """
# Basic dataset loading
from data.data_loader import load_dataset, split_data

# Load simple dataset
X, y = load_dataset('simple_linear_low_noise.csv')

# Load multivariate dataset with feature names
X, y, feature_names = load_dataset('housing_prices.csv', return_feature_names=True)

# Split data for training and testing
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42)

# Quick benchmark datasets
from data.data_loader import get_benchmark_dataset
X_small, y_small = get_benchmark_dataset('small')    # Small dataset
X_medium, y_medium = get_benchmark_dataset('medium')  # Medium dataset  
X_large, y_large = get_benchmark_dataset('large')    # Large dataset

# Load all simple datasets at once
from data.data_loader import load_simple_datasets
simple_data = load_simple_datasets()
for name, (X, y) in simple_data.items():
    print(f"{name}: {X.shape[0]} samples")

# Example usage in your linear regression implementation
from linear_regression.model import LinearRegression

# Test on multiple datasets
datasets_to_test = ['simple_linear_low_noise.csv', 'housing_prices.csv']
for dataset_name in datasets_to_test:
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"{dataset_name}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}")
"""
    
    print(examples)


def main():
    """Run all demonstration examples."""
    print("🎯 Dataset Usage Demonstrations")
    print("=" * 50)
    
    try:
        # Check if datasets exist
        datasets = list_available_datasets()
        if not datasets:
            print("❌ No datasets found! Please run 'python data/generate_datasets.py' first.")
            return
        
        print(f"✅ Found {len(datasets)} datasets")
        
        # Run demonstrations
        demonstrate_simple_dataset()
        demonstrate_multivariate_dataset()
        demonstrate_outlier_dataset()
        demonstrate_noise_comparison()
        performance_testing_datasets()
        create_usage_examples()
        
        print("\n🎉 Dataset demonstrations complete!")
        print("All visualizations have been saved as PNG files.")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
