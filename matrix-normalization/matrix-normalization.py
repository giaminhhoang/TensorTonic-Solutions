import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    # Write code here
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        return None
    if axis != None and axis >=2:
        return None
    if norm_type=='l2':
        ord=None
    elif norm_type=='l1':
        ord=1
    elif norm_type=='max':
        ord=np.inf
    else:
        return None
    norm = np.linalg.norm(matrix, ord=ord ,axis=axis, keepdims=True)
    return np.where(norm==0, 0, matrix/norm)