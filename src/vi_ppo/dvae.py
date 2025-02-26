from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence

from vi_ppo.nets.gumbel_softmax import gumbel_softmax


@dataclass
class DiscreteVaeConfig:
    beta: float = 1.0  # kl divergence coefficient
    tau: float = 1e-6
    z_dim: int = 8
    z_layers: int = 4


class DiscreteVae(nn.Module):
    config_cls = DiscreteVaeConfig

    def __init__(
        self,
        config: DiscreteVaeConfig,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.encoder(x)
        z = self.reparameterize(logits)
        x_hat = self.decoder(z)

        return x_hat, z, logits

    def reparameterize(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits.view(-1, self.config.z_layers, self.config.z_dim)

        if self.training:
            z = gumbel_softmax(logits=logits, tau=self.config.tau, hard=False)
        else:
            s = torch.argmax(logits, dim=-1)  # tensor of n_batch * self.z_n_layers
            z = F.one_hot(s, num_classes=self.z_dim)
        return z.view(-1, self.config.z_layers * self.config.z_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.reparameterize(self.encoder(x))

    def kl_divergence(self, logits):
        """
        Logits are shape (B, N * K), where B is the number of batches, N is the number
            of categorical distributions and where K is the number of classes
        # returns kl-divergence, in nats
        """

        B = logits.shape[0]
        logits = logits.view(-1, self.config.z_dim)

        q = Categorical(logits=logits)
        p = Categorical(
            probs=torch.full(
                (B * self.config.z_layers, self.config.z_dim), 1.0 / self.config.z_dim
            ).to(logits.device)
        )

        # sum loss over dimensions in each example, average over batch
        kl = kl_divergence(q, p).view(B, self.config.z_layers).sum(1).mean()

        return kl * self.config.beta

    def loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, _, latents = self(x)
        recon_loss = F.mse_loss(x_hat, x, reduction="none").sum(1).mean()
        kl_loss = self.kl_divergence(latents)
        return recon_loss + kl_loss * self.config.beta
