from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ActorCriticConfig:
    clip_epsilon: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01
    normalize_advantages: bool = True


class ActorCritic(nn.Module):
    config_cls = ActorCriticConfig

    def __init__(
        self,
        config: ActorCriticConfig,
        actor_net: nn.Module,
        critic: nn.Module,
        feature_extractor: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        self.actor = actor_net
        self.critic = critic
        self.feature_extractor = (
            feature_extractor if feature_extractor else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.feature_extractor(x)
        action_logits = self.actor(x)
        critic_values = self.critic(x)

        return action_logits, critic_values

    def sample_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self(x)
        log_probs = F.log_softmax(logits, dim=-1)
        a = torch.multinomial(log_probs.exp(), num_samples=1)
        log_prob = log_probs.gather(dim=-1, index=a).squeeze(-1)
        return a, log_prob, values

    def loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the PPO loss.

        Returns:
            torch.Tensor: The computed PPO loss.
        """
        action_logits, values = self.forward(batch["observations"])

        # Compute log probabilities of the chosen actions
        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(batch["actions"])

        # Compute ratio of new and old probabilities
        old_log_probs = batch["log_probs"]
        ratio = torch.exp(log_probs - old_log_probs)

        # Policy loss.
        advantages = batch["advantages"]
        advantages = (
            advantages
            if self.config.normalize_advantages
            else (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        )

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(
            ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
        )
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value function loss (squared error loss)
        value_loss = F.mse_loss(values.squeeze(-1), batch["returns"])

        # Entropy bonus (to encourage exploration)
        entropy = dist.entropy().mean()

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_coeff * value_loss
            - self.config.entropy_coeff * entropy
        )
        return total_loss
