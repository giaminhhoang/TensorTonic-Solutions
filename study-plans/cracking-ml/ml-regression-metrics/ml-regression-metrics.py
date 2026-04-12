import numpy as np

def regression_metrics(y_true, y_pred):
    """
    Returns: dict with keys "mse", "mae", "r2" rounded to 4 decimal places
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - mse*len(y_true)/ss_tot if ss_tot > 0.0 else 0.0
    return {"mse": round(mse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}
    