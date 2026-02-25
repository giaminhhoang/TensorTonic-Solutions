import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    num_class = y_pred.shape[1]
    class_pred = np.arange(num_class)
    label = (class_pred == y_true[:, None]).astype(int)
    return -np.mean(np.sum(label * np.log(np.clip(y_pred, 1e-12, 1.0)), axis=1))
    