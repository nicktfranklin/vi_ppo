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

    vae_loss_coeff: float = 1.0


class ActorCritic(nn.Module):
    config_cls = ActorCriticConfig

    def __init__(
        self,
        config: ActorCriticConfig,
        actor_net: nn.Module,
        critic: nn.Module,
        feature_extractor: nn.Module | None = None,
        state_vae: nn.Module | None = None,
        transition_network: nn.Module | None = None,
        action_embeddings: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config
        self.actor = actor_net
        self.critic = critic
        self.feature_extractor = (
            feature_extractor if feature_extractor else nn.Identity()
        )
        self.state_vae = state_vae  # allow for None
        self.transition_network = transition_network  # allow for None
        self.action_embeddings = action_embeddings  # allow for None

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

    def get_state(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(x)
        if self.state_vae is None:
            return features
        return self.state_vae.encode(features)

    def vae_loss(
        self, features: torch.Tensor, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Note: the VAE operaterates on the features, not raw input.  The logic of this
        is to allow for a simpler VAE architecture.

        If we want the vae to operate on the raw input, this can be done by not using a feature extractor.
        """

        if self.state_vae is None:
            return torch.tensor(0.0), {}

        vae_loss, state = self.state_vae.loss(features)
        metrics = {
            "train/vae_loss": vae_loss,
        }

        if self.transition_network is not None:
            """model takes in states and actions and predicts the next state and the reward"""

            # get the next state
            next_obs_features = self.feature_extractor(batch["next_observations"])
            _vae_loss, next_state = self.state_vae.loss(next_obs_features)
            vae_loss += _vae_loss

            action = self.action_embeddings(batch["actions"])

            y_hat = self.transition_network(torch.cat([state, action], dim=-1))

            target = torch.cat([next_state, batch["rewards"].unsqueeze(1)], dim=-1)
            transition_loss = F.mse_loss(y_hat, target)

            metrics = {
                "train/vae_loss": vae_loss,
                "train/transition_loss": transition_loss,
            }

        return vae_loss, metrics

    def loss(
        self, batch: dict[str, torch.Tensor], return_metrics: bool = False
    ) -> torch.Tensor:
        """
        Compute the PPO loss.

        Returns:
            torch.Tensor: The computed PPO loss.
        """
        features = self.feature_extractor(batch["observations"])
        action_logits = self.actor(features)
        values = self.critic(features)

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

        # VAE loss
        vae_loss, vae_metrics = self.vae_loss(features, batch)

        # Total loss
        total_loss = (
            policy_loss
            + self.config.value_coeff * value_loss
            - self.config.entropy_coeff * entropy
            + self.config.vae_loss_coeff * vae_loss
        )
        if return_metrics:
            metrics = {
                "train/value_loss": value_loss,
                "train/policy_loss": policy_loss,
                "train/loss": total_loss,
            }
            if (self.state_vae is not None) and (self.config.vae_loss_coeff > 0):
                metrics.update(vae_metrics)

            return total_loss, metrics
        return total_loss
