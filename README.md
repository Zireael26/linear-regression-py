# Linear Regression Implementation with Python

This repository contains a comprehensive implementation of linear regression from scratch using Python, NumPy, and Matplotlib. The project demonstrates how to build, train, and evaluate a linear regression model without relying on machine learning libraries like scikit-learn.

## 🎯 Project Objectives

- Implement linear regression from scratch using only NumPy
- Understand the mathematical foundations behind linear regression
- Create comprehensive tests and examples
- Visualize results with Matplotlib
- Compare performance with scikit-learn implementation

## 📁 Project Structure

```
linear-regression-py/
├── linear_regression/          # Main package
│   ├── __init__.py            # Package initialization
│   ├── model.py              # LinearRegression class
│   └── utils.py              # Utility functions
├── tests/                     # Unit tests
│   ├── __init__.py
│   └── test_linear_regression.py
├── examples/                  # Example scripts
│   ├── __init__.py
│   └── basic_example.py
├── notebooks/                 # Jupyter notebooks
│   └── linear_regression_notebook.ipynb
├── data/                      # Sample datasets
│   └── README.md
├── requirements.txt           # Project dependencies
├── .gitignore                # Git ignore file
├── pytest.ini               # Pytest configuration
└── README.md                 # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd linear-regression-py
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📊 Generated Datasets

The project includes **12 comprehensive datasets** for testing your implementation:

### Simple Linear Datasets (1D)
- **Low/Medium/High Noise**: 1,000 samples each with varying noise levels
- **Small Dataset**: 20 samples for edge case testing
- **Large Dataset**: 50,000 samples for performance testing

### Multivariate Datasets
- **5-Feature Dataset**: 1,500 samples with known coefficients
- **Housing Prices**: 2,000 samples with realistic features
- **Large Multivariate**: 5,000 samples, 10 features

### Special Cases
- **Data with Outliers**: 800 samples with 8% outliers
- **Heteroscedastic Data**: Non-constant variance
- **Time Series**: Linear trend with seasonality
- **Polynomial Data**: Degree-3 polynomial for overfitting tests

### Quick Dataset Usage
```python
from data.data_loader import load_dataset, split_data

# Load any dataset
X, y = load_dataset('simple_linear_low_noise.csv')
X_train, X_test, y_train, y_test = split_data(X, y)

# Load with feature names
X, y, names = load_dataset('housing_prices.csv', return_feature_names=True)
```

## 📊 Features to Implement

### Core Features
- [ ] Linear regression model class
- [ ] Normal equation solver
- [ ] Gradient descent optimization
- [ ] Prediction functionality
- [ ] Model evaluation metrics (R², MSE)

### Advanced Features
- [ ] Multivariate linear regression
- [ ] Feature scaling/normalization
- [ ] Ridge regression (L2 regularization)
- [ ] Cross-validation
- [ ] Learning curve visualization

### Utility Functions
- [ ] Synthetic data generation
- [ ] Data visualization
- [ ] Model comparison tools
- [ ] Performance metrics

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run tests with coverage:
```bash
pytest tests/ --cov=linear_regression
```

## 📈 Usage

### Basic Example
```python
from linear_regression.model import LinearRegression
from linear_regression.utils import generate_linear_data

# Generate sample data
X, y = generate_linear_data(n_samples=100, noise=0.1)

# Create and fit the model
model = LinearRegression()
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Evaluate the model
r2_score = model.score(X, y)
print(f"R² Score: {r2_score:.4f}")
```

### Interactive Development
Use the Jupyter notebook for interactive development:
```bash
jupyter notebook notebooks/linear_regression_notebook.ipynb
```

## 📚 Mathematical Background

### Normal Equation
The optimal weights can be calculated using:
```
θ = (X^T X)^(-1) X^T y
```

### Gradient Descent
Iterative optimization using:
```
θ := θ - α ∇J(θ)
```

Where J(θ) is the Mean Squared Error cost function.

## 🎓 Learning Goals

1. **Mathematical Understanding**: Grasp the linear algebra behind linear regression
2. **Implementation Skills**: Build algorithms from mathematical formulations
3. **Testing Practices**: Write comprehensive unit tests
4. **Code Organization**: Structure a Python project professionally
5. **Visualization**: Create meaningful plots to understand model behavior
6. **Performance Analysis**: Compare custom implementation with established libraries

## 🔧 Development Tips

- Start with the basic `LinearRegression` class in `linear_regression/model.py`
- Implement utility functions in `linear_regression/utils.py`
- Write tests as you develop features
- Use the Jupyter notebook for experimentation
- Visualize your results to verify correctness

## 📝 TODOs

- [ ] Implement core LinearRegression class
- [ ] Add data generation utilities
- [ ] Create visualization functions
- [ ] Write comprehensive tests
- [ ] Add example datasets
- [ ] Document mathematical derivations
- [ ] Create performance benchmarks

## 🤝 Contributing

This is a learning project. Feel free to:
- Add new features
- Improve documentation
- Add more test cases
- Create additional examples

## 📄 License

This project is for educational purposes.