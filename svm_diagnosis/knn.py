import numpy as np


class KNN:
    """
    A simple K-Nearest Neighbors (KNN) classifier using only NumPy.
    This can be used to classify grayscale intensity pixels into black (0) or white (1).
    """

    def __init__(self, k=5):
        self.k = k  # Number of nearest neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """
        Stores the training data (no actual training required for KNN).
        :param X_train: NumPy array of shape (n_samples, n_features)
        :param y_train: NumPy array of shape (n_samples,)
        """
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        """
        Predicts the labels for the given test data using majority voting.
        :param X_test: NumPy array of shape (n_samples, n_features)
        :return: Predicted labels of shape (n_samples,)
        """
        predictions = []
        for x in X_test:
            # Compute Euclidean distance between test point and all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)
            
            # Get indices of the k nearest neighbors
            k_nearest_indices = np.argsort(distances)[:self.k]
            
            # Get labels of the k nearest neighbors
            k_nearest_labels = self.y_train[k_nearest_indices]
            
            # Majority vote
            unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]
            
            predictions.append(majority_label)
        
        return np.array(predictions)
