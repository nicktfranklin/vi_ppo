from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VaeConfig:
    beta: float = 1.0
    sigma: float = 1e-6
    z_dim: int = 8


class Vae(nn.Module):
    config_cls = VaeConfig

    def __init__(self, config: VaeConfig, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

        self.log_var = torch.log(torch.tensor([self.config.sigma]))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.encoder(x)
        z, mu, log_var = self.reparameterize(latents)
        x_hat = self.decoder(z)

        return x_hat, z, mu, log_var

    def reparameterize(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = latents.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)

        if self.training:
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        return z, mu, log_var

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.reparameterize(self.encoder(x))

    def kl_divergence(self, mu_q, logvar_q):
        """
        Compute the KL divergence between two diagonal Gaussian distributions:

        q ~ N(mu_q, sigma_q^2) and p ~ N(mu_p, sigma_p^2)

        where logvar = log(sigma^2)

        Parameters:
            mu_q (Tensor): Mean of q.
            logvar_q (Tensor): Log-variance of q.
            mu_p (Tensor): Mean of p.
            logvar_p (Tensor): Log-variance of p.

        Returns:
            Tensor: The KL divergence summed over the dimensions.
        """
        logvar_p = self.log_var.to(logvar_q.device)
        mu_p = torch.tensor([0.0]).to(mu_q.device)

        # Calculate elementwise variances
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)

        # Compute the KL divergence term by term
        kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)

        # Sum over dimensions (if your inputs are multi-dimensional)
        return torch.sum(kl)

    def loss(self, x: torch.Tensor, target: torch.Tensor | None = None) -> torch.Tensor:
        x_hat, z, mu, log_var = self(x)
        target = x if target is None else target
        recon_loss = F.mse_loss(x_hat, target)
        kl_loss = self.kl_divergence(mu, log_var)
        return recon_loss + self.config.beta * kl_loss, z
