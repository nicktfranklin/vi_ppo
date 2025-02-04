from dataclasses import dataclass

import lightning as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from src.actor_critic import ActorCritic
from src.buffer import RolloutBuffer


@dataclass
class PPOModuleConfig:
    rollout_length: int = 2048
    update_epochs: int = 10
    lr: float = 1e-3
    gamma: float = 0.95
    clip_epsilon: float = 0.2
    value_coeff: float = 0.5
    entropy_coeff: float = 0.01


class PPOModule(pl.LightningModule):
    config_class = PPOModuleConfig

    def __init__(
        self,
        actor_critic: ActorCritic,
        env,
        config: PPOModuleConfig,
    ):
        """
        Args:
            actor_critic: Your PPO policy network.
            env: The environment instance (must follow the provided API).
            rollout_length: Number of steps to collect per rollout.
            update_epochs: Number of training epochs (full passes over the collected rollout)
                          before collecting new data.
            lr, gamma, clip_epsilon, value_coeff, entropy_coeff: PPO hyperparameters.
        """
        super().__init__()
        self.actor_critic = actor_critic
        self.env = env
        self.config = config

        # Rollout buffer should have methods to add transitions, get a dataset, and clear itself.
        self.buffer = RolloutBuffer()
        self.current_update_epoch = 0  # counts epochs since last rollout collection
        self.rollout_dataset = None

    def forward(self, x):
        return self.actor_critic(x)

    def predict(self, x: np.ndarray):
        """Predict the action to take given the current state."""
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, _, _ = self.actor_critic.sample_action(x)
        return action.item()

    def collect_rollout(self):
        """Collects a rollout using the current policy."""
        obs, info = self.env.reset()
        self.actor_critic = self.actor_critic.to(self.device)
        self.total_rewards = 0

        # Store value of final next_obs (critical for GAE)
        final_next_obs_value = 0

        for _ in range(self.config.rollout_length):
            # Convert state to tensor and add batch dimension.
            obs_tensor = (
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                action, log_prob, values = self.actor_critic.sample_action(obs_tensor)

            # Step in the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action.item())
            self.total_rewards += reward

            # Store the transition (assuming your buffer.add signature matches these arguments)
            self.buffer.add(
                obs=obs,
                act=action.item(),
                rew=reward,
                next_obs=next_obs,
                terminated=terminated,
                truncated=truncated,
                info=info,
                log_probs=log_prob.detach().clone().item(),
                values=values.detach().clone().item(),
            )

            obs = next_obs
            # Capture final next_obs value
            if _ == self.config.rollout_length - 1:
                with torch.no_grad():
                    final_next_obs_value = self.actor_critic(
                        torch.tensor(next_obs, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(self.device)
                    )[1].item()

            if terminated or truncated:
                obs, info = self.env.reset()
        self.buffer.add_terminal_values(final_next_obs_value)

    def train_dataloader(self):
        """
        Each call to train_dataloader is used to supply data for one Lightning epoch.
        We choose to collect new rollout data only every `update_epochs` epochs.
        """
        if self.current_update_epoch == 0:
            # Before collecting new rollouts, clear previous data.
            self.buffer.clear()
            self.collect_rollout()
            # Assume self.buffer.get_dataset() returns a torch Dataset yielding:
            # (observations, actions, old_log_probs, returns, advantages)
            self.rollout_dataset = self.buffer.get_dataset()

        # Increment epoch counter and reset when reaching update_epochs.
        self.current_update_epoch += 1
        if self.current_update_epoch >= self.config.update_epochs:
            self.current_update_epoch = 0

        return DataLoader(self.rollout_dataset, batch_size=64, shuffle=True)

    def compute_loss(self, batch: dict):
        """
        Unpacks the batch and computes the PPO loss.
        Expects the batch to provide:
          obs, actions, old_log_probs, returns, advantages.
        """
        # Forward pass through actor-critic.
        action_logits, values = self.actor_critic(batch["observations"])
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(batch["actions"])

        # Compute probability ratio.
        ratio = torch.exp(log_probs - batch["log_probs"])
        clipped_ratio = torch.clamp(
            ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
        )

        # Policy loss.
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        self.log("train/policy_loss", policy_loss, prog_bar=False, logger=True)

        # Value loss.
        value_loss = nn.functional.mse_loss(values.squeeze(), batch["returns"])
        self.log("train/value_loss", value_loss, prog_bar=False, logger=True)

        # Entropy bonus.
        entropy_bonus = dist.entropy().mean()

        total_loss = (
            policy_loss
            + self.config.value_coeff * value_loss
            - self.config.entropy_coeff * entropy_bonus
        )
        self.log("train/loss", total_loss, prog_bar=False, logger=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        return loss

    def on_train_epoch_start(self):
        """Log the total reward of the rollout at the start of the first training epoch using it."""
        if self.current_update_epoch == 1:
            self.log(
                "train/total_reward", self.total_rewards, prog_bar=True, logger=True
            )

    def configure_optimizers(self):
        return optim.Adam(self.actor_critic.parameters(), lr=self.config.lr)
