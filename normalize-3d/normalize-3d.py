import numpy as np

def normalize_3d(v):
    """
    Normalize 3D vector(s) to unit length.
    """
    # Your code here
    v = np.atleast_2d(np.asarray(v, dtype=float))
    norm = np.sqrt(np.sum(v * v, axis=1, keepdims=True))
    norm = np.where(norm >= 1.0e-10, norm, 1.0)
    return (v / norm).squeeze()