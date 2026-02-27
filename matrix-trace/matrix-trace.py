import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here
    A = np.asarray(A, dtype=float)
    rows = A.shape[0]

    trA = 0.0
    for r in range(rows):
        trA += A[r][r]
    return trA
