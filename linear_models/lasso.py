import numpy as np

class LassoWithSoftThreshold:
    def __init__(self, alpha=1.0, n_iter=1000, lr=0.01):
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = 0.0

    def _soft_threshold(self, x, gamma):
        return np.sign(x) * np.maximum(np.abs(x) - gamma, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.ones(n_features) * 0.01

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            w_temp = self.weights - self.lr*dw
            self.weights = self._soft_threshold(w_temp, self.lr * self.alpha)
            self.bias -= self.lr*db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred

import numpy as np

class LassoWithSubgradient:
    def __init__(self, alpha=1.0, n_iter=1000, lr=0.01):
        self.alpha = alpha
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.ones(n_features) * 0.01

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + self.alpha * np.sign(self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights = self.weights - self.lr*dw
            self.bias -= self.lr*db

    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred