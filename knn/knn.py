"""
K-Nearest Neighbors (KNN) Classifier and Regressor (from scratch)

KNN is a lazy, non-parametric algorithm used for classification and regression.
It makes predictions based on the labels/values of the 'k' closest training samples
using a specified distance metric (Euclidean here).

Key Features:
- Works well with small to medium-sized datasets.
- No training time; all computation happens at prediction time.
- Can be used for both classification and regression by changing the voting logic.

Attributes:
    k (int): Number of nearest neighbors to consider.
    is_classifier (bool): Whether the model is for classification or regression.
"""

import numpy as np
from collections import Counter

def euclidean(a,b):
    """
    Computes the Euclidean distance between two vectors.

    Parameters:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        float: Euclidean distance.
    """
    return np.sqrt(np.sum((a-b)**2))
    
class KNN:
    def __init__(self, k=3, is_classifier=True):
        """
        Initializes the KNN model.

        Parameters:
            k (int): Number of neighbors to consider.
            is_classifier (bool): Whether the model is used for classification.
        """
        self.k = k
        self.is_classifier = is_classifier

    def fit(self, X, y):
        """
        Stores the training data.

        Parameters:
            X (np.ndarray): Training features (n_samples x n_features).
            y (np.ndarray): Training labels or values (n_samples,).
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _predict(self, x):
        """
        Predicts the label/value for a single sample.

        Parameters:
            x (np.ndarray): Input vector.

        Returns:
            Predicted class (int/str) or value (float).
        """
        # Compute distances from the input point to all training samples
        distances = [euclidean(x, x_train) for x_train in self.X_train]
        
        # Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract their labels or values
        k_nearest_labels = self.y_train[k_indices]

        if self.is_classifier:
            # Return the most common label
            return Counter(k_nearest_labels).most_common(1)[0][0]
        
        else:
            return np.mean(k_nearest_labels)
        
    def predict(self, X):
        """
        Predicts labels/values for multiple samples.

        Parameters:
            X (np.ndarray): Test data (n_samples x n_features).

        Returns:
            np.ndarray: Predictions.
        """
        return np.array([self._predict(x) for x in X])