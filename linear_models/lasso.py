"""
Lasso Regression (Gradient-based with Soft-Thresholding)

The Lasso algorithm (Least Absolute Shrinkage and Selection Operator) is a linear regression
model that includes an L1 regularization term to:
- Prevent overfitting
- Automatically perform feature selection by zeroing out less important coefficients
- Improve model interpretability on sparse, high-dimensional data

This implementation uses soft-thresholding and a coordinate descent–like update strategy.

Attributes:
    coef_ (np.ndarray): Learned weights for each feature including bias.
"""

import numpy as np

class Lasso:
    def __init__(self, alpha=1.0, n_iter=1000):
        """
        Initialize the Lasso regressor.

        Parameters:
            alpha (float): Regularization strength (larger = more feature elimination).
            lr (float): Learning rate (not directly used here, kept for extensibility).
            n_iter (int): Number of iterations for optimization.
        """
        self.alpha = alpha
        self.n_iter = n_iter

    def soft_threshold(self, rho, lambda_):
        """
        Apply soft-thresholding (used for L1 penalty shrinkage).

        Parameters:
            rho (float): The unregularized coefficient update (gradient component).
            lambda_ (float): Regularization strength (L1 part).

        Returns:
            float: Thresholded value for coefficient update.
        """
        if rho < -lambda_:
            return rho + lambda_
        elif rho > lambda_:
            return rho - lambda_
        else:
            return 0.0

    def fit(self, X, y):
        """
        Fit the Lasso model to training data using coordinate-wise updates.

        Parameters:
            X (np.ndarray): Feature matrix of shape (n_samples, n_features).
            y (np.ndarray): Target vector of shape (n_samples,).
        """
        # Add bias term (intercept)
        X = np.c_[np.ones(X.shape[0]), X]
        n_samples, n_features = X.shape

        # Initialize all coefficients (including bias) to 0
        self.coef_ = np.zeros(n_features)

        for _ in range(self.n_iter):
            for i in range(n_features):
                X_j = X[:, i]
                y_pred = X @ self.coef_
                residual = y - y_pred + self.coef_[i] * X_j
                rho = np.dot(X_j, residual)

                if i == 0:
                    # Don't regularize the intercept
                    self.coef_[i] = rho / np.sum(X_j ** 2)
                else:
                    # Apply soft-thresholding for L1 penalty
                    self.coef_[i] = self.soft_threshold(rho, self.alpha) / np.sum(X_j ** 2)

    def predict(self, X):
        """
        Predict target values for new data.

        Parameters:
            X (np.ndarray): Input features (n_samples, n_features).

        Returns:
            np.ndarray: Predicted target values.
        """
        X = np.c_[np.ones(X.shape[0]), X]
        return X @ self.coef_
