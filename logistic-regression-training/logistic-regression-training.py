import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0
    for i in range(steps):
        z = X @ w + b
        p = _sigmoid(z)
        w_grad = X.T @ (p - y) / n_samples
        b_grad = np.sum(p - y) / n_samples
        w -= lr * w_grad
        b -= lr * b_grad
    return w, b