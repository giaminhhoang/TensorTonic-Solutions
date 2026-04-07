import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    w = np.zeros(X.shape[1])
    b = 0.0
    for epoch in range(epochs):
        for i in range(len(y)):
            z = np.dot(w, X[i,:]) + b
            y_hat = 1.0 if z >= 0.0 else 0.0
            w+=lr*(y[i] - y_hat)*X[i,:]
            b+=lr*(y[i] - y_hat)
    return w, b