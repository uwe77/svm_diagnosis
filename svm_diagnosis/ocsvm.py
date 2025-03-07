import numpy as np
import cvxopt

class OneClassSVM_QP:
    def __init__(self, kernel="rbf", nu=0.1, gamma=0.1):
        self.nu = nu
        self.gamma = gamma
        self.kernel = self._get_kernel_function(kernel)
        self.alpha = None
        self.support_vectors = None
        self.decision_boundary = None

    def _get_kernel_function(self, kernel_type):
        if kernel_type == "rbf":
            return lambda x, y: np.exp(-self.gamma * np.linalg.norm(x - y, axis=1) ** 2)
        elif kernel_type == "linear":
            return lambda x, y: np.dot(x, y.T)
        else:
            raise ValueError("Unsupported kernel type. Use 'rbf' or 'linear'.")

    def fit(self, X):
        n_samples = X.shape[0]

        # Compute Kernel Matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            K[i] = self.kernel(X[i], X)

        # Define Quadratic Programming problem
        P = cvxopt.matrix(K)
        q = cvxopt.matrix(-np.ones((n_samples, 1)))
        G = cvxopt.matrix(np.vstack([-np.eye(n_samples), np.eye(n_samples)]))
        h = cvxopt.matrix(np.hstack([np.zeros(n_samples), np.ones(n_samples) / (self.nu * n_samples)]))
        A = cvxopt.matrix(np.ones((1, n_samples)), tc='d')
        b = cvxopt.matrix(np.array([1.0]))

        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(solution['x'])

        # Get support vectors
        support_mask = self.alpha > 1e-6
        self.support_vectors = X[support_mask]
        self.alpha = self.alpha[support_mask]

        # Compute decision boundary (ρ)
        self.decision_boundary = np.mean(np.dot(K[support_mask][:, support_mask], self.alpha))

    def decision_function(self, X):
        """Compute the decision function values for X."""
        scores = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            scores[i] = np.sum(self.alpha * self.kernel(X[i], self.support_vectors))
        return scores - self.decision_boundary

    def predict(self, X):
        """Predict if X is normal (1) or an outlier (-1)."""
        return np.sign(self.decision_function(X))
