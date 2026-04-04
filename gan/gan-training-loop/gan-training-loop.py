import numpy as np

def train_gan_step(real_data: np.ndarray, generator, discriminator, noise_dim: int) -> dict:
        """
            Perform one training step for GAN.
                """""
        # Step 1: Train Discriminator
        z = np.random.randn(noise_dim)
        fake_data = generator.generate(z)
        d_loss = discriminator.train_on_batch(real_data, fake_data)

        # Step 2: Train Generator (new noise to avoid correlation)
        z = np.random.randn(noise_dim)
        g_loss = generator.train_on_batch(discriminator, z)

        return {"d_loss": d_loss, "g_loss": g_loss}