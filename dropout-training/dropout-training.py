import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.asarray(x, dtype=float)
    if rng is not None:
        mask = rng.random(x.shape) < 1 - p
    else:
        mask = np.random.rand(*x.shape) < 1 - p
    mask = mask * 1 / (1 - p)
    return x * mask, mask
    