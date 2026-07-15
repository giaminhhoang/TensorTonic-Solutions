import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # delta = np.float(delta)
    err =  y_pred - y_true
    idx1 = np.where(np.abs(err) <= delta)
    idx2 = np.where(np.abs(err) > delta)
    L = np.zeros_like(y_pred)
    L[idx1] =  0.5*err[idx1]**2
    L[idx2] = delta*(np.abs(err[idx2]) - 0.5*delta)
    return np.mean(L)