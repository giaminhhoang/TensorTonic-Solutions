import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p, dtype=float).flatten()
    y = np.asarray(y, dtype=float).flatten()

    dice = (2.0*np.sum(p * y) + eps)/(np.sum(p) + np.sum(y) + eps)
    return 1 - dice