import numpy as np

def dice_loss(p, y, eps=1e-8):
    """
    Compute Dice Loss for segmentation.
    """
    # Write code here
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)

    if p.ndim == 2:
        p = p.flatten()
        y = y.flatten()

    dice = (2.0*np.sum(p * y) + eps)/(sum(p) + sum(y) + eps)
    return 1 - dice