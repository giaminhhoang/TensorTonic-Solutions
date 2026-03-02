import numpy as np

def generator(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Generate fake data from noise vectors.
    """
    # Your implementation here
    W = np.random.randn(z.shape[1], output_dim) * 0.01
    b = np.random.randn(output_dim) * 0.01
    x_hat = np.tanh(z @ W + b)
    return x_hat