"""
Linear Regression Model Implementation

This module contains the LinearRegression class that implements
linear regression from scratch using numpy.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class LinearRegression:
    """
    Linear Regression implementation from scratch using numpy.
    
    This class provides methods to fit a linear model to data,
    make predictions, and evaluate model performance.
    
    Attributes:
        weights (np.ndarray): Model weights (coefficients)
        bias (float): Model bias (intercept)
        fitted (bool): Whether the model has been fitted to data
    """
    
    def __init__(self):
        """Initialize the Linear Regression model."""
        # TODO: Implement initialization
        pass
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the linear regression model to the training data.
        
        Args:
            X (np.ndarray): Training features of shape (n_samples, n_features)
            y (np.ndarray): Training targets of shape (n_samples,)
        """
        # TODO: Implement model fitting
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted model.
        
        Args:
            X (np.ndarray): Input features of shape (n_samples, n_features)
            
        Returns:
            np.ndarray: Predictions of shape (n_samples,)
        """
        # TODO: Implement prediction
        pass
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate the R² score of the model.
        
        Args:
            X (np.ndarray): Test features
            y (np.ndarray): True targets
            
        Returns:
            float: R² score
        """
        # TODO: Implement R² score calculation
        pass
