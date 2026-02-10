import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    else:
        return np.exp(x - np.max(x, axis=1)[:,None]) / np.sum(np.exp(x - np.max(x, axis=1)[:,None]), axis=1)[:,None]