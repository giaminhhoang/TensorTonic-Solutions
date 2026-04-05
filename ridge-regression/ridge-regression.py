import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    return np.linalg.inv(X.T @ X + lam*np.eye(X.shape[1])) @ X.T @ y