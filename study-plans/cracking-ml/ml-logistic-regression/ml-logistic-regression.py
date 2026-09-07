import numpy as np

def logistic_regression(X, y, lr=0.01, n_iters=1000):
    """
    Returns:
        tuple: (weights, bias) where weights is a list and bias is a float
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = np.shape(X)
    w = np.zeros(d)
    b = 0.0
    for _ in range(n_iters):
        z = X @ w + b
        y_hat = 1 / (1 + np.exp(-z))
        grad_w = X.T @ (y_hat - y) / n
        grad_b = np.sum(y_hat - y) / n
        w = w - lr * grad_w
        b = b - lr * grad_b
    return w, b