"""
Ridge Regression from Scratch

Ridge Regression is a linear model that includes L2 regularization to prevent
overfitting and handle multicollinearity (high correlation among independent features).

It’s ideal for:
- Dense datasets
- Situations where all features might be relevant
- Reducing model variance at the cost of a small increase in bias

This module includes:
- Ridge: Gradient Descent–based implementation (scalable)
- RidgeClosed: Closed-form solution (exact, but expensive for large datasets)
"""

import numpy as np

class Ridge:
    """
    Ridge Regression using Gradient Descent (L2 regularization).

    Attributes:
        coef_ (np.ndarray): Learned weights including the bias term.
    """
    def __init__(self, alpha=0.1, lr=0.01, n_iter=1000):
        """
        Initialize model hyperparameters.

        Parameters:
            alpha (float): Regularization strength.
            lr (float): Learning rate for gradient descent.
            n_iter (int): Number of iterations.
        """
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter

    def fit(self, X, y):
        """
        Fit Ridge model to data using gradient descent.

        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features).
            y (np.ndarray): Target vector (n_samples,).
        """
        X = np.c_[np.ones(X.shape[0]), X]  # Add bias term
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)

        for _ in range(self.n_iter):
            y_pred = X @ self.coef_
            gradient = (1 / n_samples) * X.T @ (y_pred - y)
            gradient[1:] += self.alpha * self.coef_[1:]  # Don't regularize bias
            self.coef_ -= self.lr * gradient

    def predict(self, X):
        """
        Predict target values using trained Ridge model.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted values.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef_


class RidgeClosed:
    """
    Ridge Regression using Closed-Form Solution (Normal Equation).

    Attributes:
        coef_ (np.ndarray): Optimal weights including the bias term.
    """
    def __init__(self, alpha):
        """
        Initialize model with regularization strength.

        Parameters:
            alpha (float): Regularization strength.
        """
        self.alpha = alpha

    def fit(self, X, y):
        """
        Fit model using the closed-form solution:

        θ = (XᵀX + αI)^(-1) Xᵀy

        Bias term is not regularized.

        Parameters:
            X (np.ndarray): Feature matrix (n_samples, n_features).
            y (np.ndarray): Target vector.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        n_features = X.shape[1]
        I = np.eye(n_features)
        I[0, 0] = 0  # Don’t penalize bias term

        self.coef_ = np.linalg.inv(X.T @ X + self.alpha * I) @ X.T @ y

    def predict(self, X):
        """
        Predict values using the learned weights.

        Parameters:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predictions.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef_