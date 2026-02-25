import numpy as np

def cross_entropy_loss(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred)
    n = y_pred.shape[0]
    log_probs = np.log(np.clip(y_pred, 1e-12, 1.0))
    return -np.mean(log_probs[np.arange(n), y_true])
    