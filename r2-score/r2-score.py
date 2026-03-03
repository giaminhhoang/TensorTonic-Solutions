import numpy as np

def r2_score(y_true, y_pred) -> float:
    """
    Compute R² (coefficient of determination) for 1D regression.
    Handle the constant-target edge case:
      - return 1.0 if predictions match exactly,
      - else 0.0.
    """
    # Write code here
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    
    SS_total = np.sum((y_true - np.mean(y_true))**2)
    SS_res = np.sum((y_pred - y_true)**2)

    # corner case
    if SS_total == 0.0:
        return 1.0 if all(y_pred == y_true) else 0.0
    else:
        return float(1.0 - SS_res/SS_total)