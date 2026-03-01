import numpy as np

class VAE:
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Initialize VAE.
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Initialize weights here
        self.W_mu = np.random.randn(input_dim, latent_dim)
        self.W_logvar = np.random.randn(input_dim, latent_dim)
        self.W_dec = np.random.randn(latent_dim, input_dim)
    
    def forward(self, x: np.ndarray) -> tuple:
        """
        Full forward pass through VAE.
        """
        # Your implementation here
        self.mu = x@self.W_mu
        self.logvar = x@self.W_logvar
        x_hat = self.generate(x.shape[0])
        return (x_hat, self.mu, self.logvar)
    
    def generate(self, n_samples: int) -> np.ndarray:
        """
        Generate new samples from prior.
        """
        # Your implementation here
        # z = self.mu + np.exp(0.5*self.logvar)*np.random.randn(n_samples, self.latent_dim)
        z = np.random.randn(n_samples, self.latent_dim)
        return z@self.W_dec