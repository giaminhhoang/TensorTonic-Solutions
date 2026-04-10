import numpy as np

def linear_regression(X, y, lr, epochs):
    """
    Returns: tuple (weights, bias)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        y_hat = X@w + b
        w -= lr * 2/n*X.T@(y_hat - y)
        b -= lr * 2/n*np.sum(y_hat-y)

    weights = [round(float(v), 4) for v in w]
    bias = round(float(b), 4)
    return (weights, bias)
