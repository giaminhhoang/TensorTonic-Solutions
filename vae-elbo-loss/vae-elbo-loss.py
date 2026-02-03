import numpy as np

def kl_divergence(mu: np.ndarray, log_var: np.ndarray) -> float:
    """
    Compute KL divergence between q(z|x) and N(0, I).
    """
    # Your implementation here
    return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))

def vae_loss(x: np.ndarray, x_recon: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> dict:
    """
    Compute VAE ELBO loss.
    """
    # Your implementation here
    recon = np.sum((x - x_recon)**2)
    kl = kl_divergence(mu, log_var)
    return {"total": recon + kl, "recon": recon, "kl": kl}