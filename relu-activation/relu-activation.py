import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = x if isinstance(x, list) else [x]
    return np.maximum(0.0, x)