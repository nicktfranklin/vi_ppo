import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int,
        n_layers: int,
        activation: str = "silu",
    ):
        super().__init__()
        self.input_projection = (
            nn.Linear(input_dims, hidden_dims)
            if (input_dims != hidden_dims)
            else nn.Identity()
        )
        self.output_projection = (
            nn.Linear(hidden_dims, output_dims)
            if (hidden_dims != output_dims)
            else nn.Identity()
        )

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dims, hidden_dims) for _ in range(n_layers)]
        )

        if activation == "silu":
            self.act_fn = nn.SiLU()
        elif activation == "relu":
            self.act_fn = nn.ReLU()
        elif activation == "tanh":
            self.act_fn = nn.Tanh()
        else:
            raise ValueError("only silu is supported")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        for layer in self.hidden_layers:
            x = self.act_fn(layer(x)) + x
        x = self.output_projection(x)
        return x


class ActorCritic(nn.Module):
    def __init__(
        self,
        actor_net: nn.Module,
        critic: nn.Module,
        feature_extractor: nn.Module | None = None,
    ):
        super().__init__()
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

    def loss(
        self,
        observation: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon=0.2,
        value_coeff=0.5,
        entropy_coeff=0.01,
    ):
        """
        Compute the PPO loss.

        Args:
            states (torch.Tensor): Batch of states.
            actions (torch.Tensor): Batch of actions taken.
            old_log_probs (torch.Tensor): Log probabilities of actions under the old policy.
            returns (torch.Tensor): Target values for the critic.
            advantages (torch.Tensor): Computed advantage estimates.
            clip_epsilon (float): Clipping parameter for PPO.
            value_coeff (float): Coefficient for the value loss.
            entropy_coeff (float): Coefficient for the entropy bonus.

        Returns:
            torch.Tensor: The computed PPO loss.
        """
        action_logits, values = self.forward(observation)

        # Compute log probabilities of the chosen actions
        dist = torch.distributions.Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)

        # Compute ratio of new and old probabilities
        ratio = torch.exp(log_probs - old_log_probs)

        # Clipped surrogate objective
        clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value function loss (squared error loss)
        value_loss = F.mse_loss(values.squeeze(-1), returns)

        # Entropy bonus (to encourage exploration)
        entropy = dist.entropy().mean()

        # Total loss
        total_loss = policy_loss + value_coeff * value_loss - entropy_coeff * entropy
        return total_loss
