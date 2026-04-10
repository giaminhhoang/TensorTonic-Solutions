import numpy as np

def activation_functions(x, activation):
    """
    Returns: list
    """
    x = np.asarray(x, dtype=np.float64)
    
    match activation:
        case "relu":
            z = max(0.0, x)
            grad = 1.0 if x > 0.0 else 0.0
        case "sigmoid":        
            z = 1.0 / (1 + np.exp(-x))
            grad = z*(1.0-z)
        case "leaky_relu":
            z = x if x > 0.0 else 0.01*x
            grad = 1.0 if x > 0.0 else 0.01
        case "tanh":
            z = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
            grad = 1 - z**2
        case "swish":
            sig = 1.0 / (1 + np.exp(-x))
            z = x*sig
            grad = sig + x*sig*(1.0-sig)
        case "gelu":
            u = np.sqrt(2.0/np.pi)*(x+0.044715*x**3)
            t = np.tanh(u)
            v = np.sqrt(2.0/np.pi)*(1.0+3.0*0.044715*x**2)
            z = 0.5*x*(1+t)
            grad = 0.5*(1+t)+0.5*x*(1-t**2)*v
    return [z, grad]