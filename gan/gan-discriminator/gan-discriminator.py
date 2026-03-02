import numpy as np

def discriminator(x: np.ndarray) -> np.ndarray:
    """
    Classify inputs as real or fake.
    """
    # Your implementation here
    W = np.random.randn(x.shape[1], 1) * 0.01
    b = np.random.randn(1) * 0.01

    logits = x @ W + b
    proba = 1 / (1 + np.exp(-logits))
    return proba