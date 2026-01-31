import numpy as np

def vae_decoder(z: np.ndarray, output_dim: int) -> np.ndarray:
    """
    Decode latent vectors to reconstructed data.
    """
    # Your implementation here
    latent_dim = z.shape[1]
    W = np.random.randn(latent_dim, output_dim) * 0.01
    x_hat = z @ W
    x_hat = 1 / (1 + np.exp(-x_hat))
    return x_hat