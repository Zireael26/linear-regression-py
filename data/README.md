# 📊 Datasets for Linear Regression Testing

This directory contains comprehensive datasets designed to test various aspects of your linear regression implementation. The datasets range from simple 1D problems to complex multivariate scenarios with different characteristics.

## 🎯 Dataset Categories

### Simple Linear Datasets (1D)
Perfect for initial testing and understanding basic linear regression:

| Dataset | Samples | Features | Description |
|---------|---------|-----------|-------------|
| `simple_linear_low_noise.csv` | 1,000 | 1 | Clean linear relationship (noise std = 0.1) |
| `simple_linear_medium_noise.csv` | 1,000 | 1 | Moderate noise (noise std = 0.5) |
| `simple_linear_high_noise.csv` | 1,000 | 1 | High noise (noise std = 1.0) |
| `small_dataset.csv` | 20 | 1 | Tiny dataset for edge case testing |

### Multivariate Datasets
Test your implementation on multiple features:

| Dataset | Samples | Features | Description |
|---------|---------|-----------|-------------|
| `multivariate_5features.csv` | 1,500 | 5 | Standard multivariate with known coefficients |
| `housing_prices.csv` | 2,000 | 5 | Realistic housing price prediction |
| `large_multivariate.csv` | 5,000 | 10 | Large dataset for performance testing |

### Special Case Datasets
Test robustness and edge cases:

| Dataset | Samples | Features | Description |
|---------|---------|-----------|-------------|
| `data_with_outliers.csv` | 800 | 1 | Linear data with 8% outliers |
| `heteroscedastic_data.csv` | 1,000 | 1 | Non-constant variance (heteroscedasticity) |
| `time_series_trend.csv` | 1,000 | 1 | Time series with linear trend + seasonality |
| `polynomial_degree3.csv` | 600 | 1 | Polynomial relationship (for overfitting tests) |

### Performance Testing
| Dataset | Samples | Features | Description |
|---------|---------|-----------|-------------|
| `large_simple_linear.csv` | 50,000 | 1 | Large dataset for speed benchmarking |

## 🚀 Quick Start

### Loading Datasets
```python
from data.data_loader import load_dataset, split_data

# Load a simple dataset
X, y = load_dataset('simple_linear_low_noise.csv')

# Load with feature names
X, y, names = load_dataset('housing_prices.csv', return_feature_names=True)

# Split into train/test sets
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
```

### Dataset Information
```python
from data.data_loader import print_dataset_summary, get_dataset_info

# Print detailed summary
print_dataset_summary('housing_prices.csv')

# Get metadata
info = get_dataset_info('simple_linear_low_noise.csv')
print(f"True slope: {info['true_slope']}")
```

### Batch Loading
```python
from data.data_loader import load_simple_datasets, load_multivariate_datasets

# Load all simple datasets
simple_data = load_simple_datasets()
for name, (X, y) in simple_data.items():
    print(f"{name}: {X.shape}")

# Load all multivariate datasets
multivariate_data = load_multivariate_datasets()
```

## 📈 Testing Your Implementation

### Progressive Testing Strategy

1. **Start Simple**: Begin with `simple_linear_low_noise.csv`
   - Perfect for debugging your basic implementation
   - Known true parameters: slope=2.5, intercept=1.0

2. **Add Complexity**: Test on multivariate data
   - `multivariate_5features.csv` has known coefficients
   - Compare your results with the true values

3. **Robustness Testing**: Use challenging datasets
   - `data_with_outliers.csv` tests robustness
   - `heteroscedastic_data.csv` tests assumptions

4. **Performance Testing**: Benchmark on large datasets
   - `large_simple_linear.csv` for speed testing
   - `large_multivariate.csv` for memory usage

### Example Testing Loop
```python
from linear_regression.model import LinearRegression

test_datasets = [
    'simple_linear_low_noise.csv',
    'multivariate_5features.csv', 
    'housing_prices.csv',
    'data_with_outliers.csv'
]

for dataset_name in test_datasets:
    print(f"\nTesting on {dataset_name}")
    X, y = load_dataset(dataset_name)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)
    
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
```

## 🎓 Learning Objectives

Each dataset type helps you learn different aspects:

- **Simple Linear**: Basic algorithm implementation
- **Multivariate**: Matrix operations and multiple features
- **Noisy Data**: Understanding model performance under noise
- **Outliers**: Robustness and data preprocessing
- **Large Datasets**: Performance optimization
- **Heteroscedastic**: Understanding model assumptions

## 📊 Dataset Generation

To regenerate all datasets:
```bash
python data/generate_datasets.py
```

To see dataset examples and visualizations:
```bash
python examples/dataset_examples.py
```

## 🔍 Metadata Format

Each CSV file includes metadata in comment lines:
```csv
# Dataset: simple_linear_low_noise.csv
# Samples: 1000
# Features: 1
# noise_std: 0.1
# true_slope: 2.5
# true_intercept: 1.0
#
X,y
-9.87,data...
```

## 📝 File Formats

- **Simple datasets**: Columns `X`, `y`
- **Multivariate datasets**: Feature columns + `target` column
- **All files**: CSV format with comment metadata
- **Feature names**: Descriptive names for multivariate datasets

## 🤝 Adding New Datasets

To add your own datasets:

1. Follow the naming convention: `descriptive_name.csv`
2. Include metadata comments at the top
3. Use either `X,y` or `feature_names...,target` column format
4. Update the data loader if needed

## 📄 Data Sources

- **Synthetic datasets**: Generated programmatically for controlled testing
- **Housing data**: Realistic but synthetic housing price data
- **All datasets**: Created for educational purposes, no real personal data
