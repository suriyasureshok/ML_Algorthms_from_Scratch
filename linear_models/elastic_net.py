import numpy as np

class ElasticNet:
    def __init__(self, l1_ratio=0.5, n_iter=1000, alpha=1.0, tol=1e-4):
        """
        Initializes the ElasticNet regressor.

        Parameters:
            l1_ratio (float): The mixing ratio between L1 and L2 (0 = Ridge, 1 = Lasso).
            n_iter (int): Maximum number of iterations for coordinate descent.
            alpha (float): Regularization strength.
            tol (float): Convergence threshold.
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.n_iter = n_iter
        self.tol = tol

    def soft_threshold(self, rho, lambda_):
        """
        Soft-thresholding operator used for L1 regularization.

        Parameters:
            rho (float): Raw weight update value.
            lambda_ (float): L1 regularization strength.

        Returns:
            float: Updated coefficient after applying soft threshold.
        """
        if rho < -lambda_:
            return rho + lambda_
        elif rho > lambda_:
            return rho - lambda_
        else:
            return 0.0

    def fit(self, X, y):
        """
        Fits the ElasticNet model to the training data using coordinate descent.

        Parameters:
            X (np.ndarray): Input features (n_samples x n_features).
            y (np.ndarray): Target values (n_samples,).
        """
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        # Center the data (removes the need to handle intercept inside the loop)
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        X_centered = X - X_mean
        y_centered = y - y_mean

        for _ in range(self.n_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                # Calculate partial residual (excluding current feature j)
                residual = y - (X @ self.coef_) + X[:, j] * self.coef_[j]
                rho = np.dot(X[:, j], residual)

                lambda_l1 = self.alpha * self.l1_ratio
                lambda_l2 = self.alpha * (1 - self.l1_ratio)

                # Update coefficient using soft-thresholding for L1 and scaling for L2
                self.coef_[j] = self.soft_threshold(rho / n_samples, lambda_l1) / (1 + lambda_l2)

            # Convergence check
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        # Recalculate intercept using uncentered data
        self.intercept_ = y_mean - np.dot(X_mean, self.coef_)

    def predict(self, X):
        return X @ self.coef_ + self.intercept_
