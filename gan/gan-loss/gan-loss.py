import numpy as np

def discriminator_loss(real_probs: np.ndarray, fake_probs: np.ndarray) -> float:
    """
    Compute discriminator loss.
    """
    # Your implementation here
    return np.mean(-np.log(np.clip(real_probs, 1e-8, 1.0)) - np.log(np.clip(1.0 - fake_probs, 1e-8, 1.0)))

def generator_loss(fake_probs: np.ndarray) -> float:
    """
    Compute generator loss.
    """
    # Your implementation here
    return np.mean(-np.log(np.clip(fake_probs, 1e-8, 1.0)))