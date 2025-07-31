"""
Linear Regression from Scratch

This module implements both:
- Gradient Descent–based Linear Regression (for large datasets)
- Closed-form Linear Regression (for smaller datasets, < 5000 samples)

Linear regression assumes:
- Linearity between features and target
- No multicollinearity (independent features shouldn't be highly correlated)
- Homoscedasticity and normally distributed errors (for inference purposes)

Classes:
    LinearRegression       - Uses Gradient Descent (Scalable)
    LinearRegressionClosed - Uses Closed-form solution (Exact but memory-heavy)
"""

import numpy as np

class LinearRegression:
    """
    Linear Regression using Gradient Descent.

    Best for large datasets (>5000 samples) where the closed-form solution
    becomes computationally expensive.

    Attributes:
        coef_ (np.ndarray): Learned weight vector including bias term.
    """
    def __init__(self, learning_rate=0.001, num_iterations=1000):
        """
        Initialize model hyperparameters.

        Parameters:
            learning_rate (float): Step size for each update.
            num_iterations (int): Number of iterations for gradient descent.
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations

    def fit(self, X, y):
        """
        Fit the model using gradient descent optimization.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        n_samples, n_features = X.shape

        # Initialize weights to zero
        self.coef_ = np.zeros(n_features)

        # Perform gradient descent updates
        for _ in range(self.num_iterations):
            y_pred = X @ self.coef_
            grad = (1 / n_features) * X.T @ (y_pred - y)
            self.coef_ -= self.learning_rate * grad

    def predict(self, X):
        """
        Predict target values using the trained model.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef_


class LinearRegressionClosed:
    """
    Linear Regression using the Closed-form (Normal Equation).

    Best for small datasets (<= 5000 samples) due to computational overhead
    of matrix inversion.

    Attributes:
        coef_ (np.ndarray): Exact solution to the linear regression problem.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Fit the model using the closed-form equation:

        θ = (XᵀX)^(-1) Xᵀy

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        X = np.c_[np.ones(X.shape[0]), X]
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        """
        Predict target values using the fitted model.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
            np.ndarray: Predicted values.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef_
