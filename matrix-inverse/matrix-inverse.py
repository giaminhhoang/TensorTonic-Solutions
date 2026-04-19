import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A, dtype=np.float64)
    if A.ndim == 2 and (A.shape[0] == A.shape[1]):
        if np.abs(np.linalg.det(A)) > 1e-10: 
            return np.linalg.inv(A)
        else:
            return None
    else:
        return None