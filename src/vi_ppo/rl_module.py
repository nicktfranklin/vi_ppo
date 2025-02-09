from dataclasses import dataclass

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from vi_ppo.actor_critic import ActorCritic
from vi_ppo.buffer import RolloutBuffer


@dataclass
class RlModuleConfig:
    rollout_length: int = 2048
    update_epochs: int = 10
    lr: float = 1e-3
    gamma: float = 0.95
    grad_clip: float = 1.0


class RlModule(pl.LightningModule):
    config_class = RlModuleConfig

    def __init__(
        self,
        actor_critic: ActorCritic,
        env,
        config: RlModuleConfig,
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

        self.automatic_optimization = False

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

        for _ in range(self.config.rollout_length):
            # Convert state to tensor and add batch dimension.
            obs_tensor = (
                torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            )

            with torch.no_grad():
                action, log_prob, values = self.actor_critic.sample_action(obs_tensor)

            # Step in the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action.item())

            with torch.no_grad():
                # compute the next value
                next_obs_tensor = (
                    torch.tensor(next_obs, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                _, next_value = self.actor_critic(next_obs_tensor)

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
                next_values=next_value.detach().clone().item(),
            )

            obs = next_obs

            if terminated or truncated:
                obs, info = self.env.reset()

    def train_dataloader(self):
        x = torch.randn(1, 1)
        return DataLoader(x, batch_size=1)

    def get_new_rollout(self, clear_buffer=True):
        """
        Each call to train_dataloader is used to supply data for one Lightning epoch.
        We choose to collect new rollout data only every `update_epochs` epochs.
        """
        if clear_buffer:
            # Before collecting new rollouts, clear previous data.
            self.buffer.clear()
        self.collect_rollout()
        # Assume self.buffer.get_dataset() returns a torch Dataset yielding:
        # (observations, actions, old_log_probs, returns, advantages)
        self.rollout_dataset = self.buffer.get_dataset()
        # self.total_reward = self.rollout_dataset.rewards.sum()
        self.log_dict(
            {"train/total_reward": self.rollout_dataset.rewards.sum()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

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
        entropy_bonus = dist.entropy().mean()  # Entropy bonus.

        # Policy loss.
        advantages = batch["advantages"]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute probability ratio.
        ratio = torch.exp(log_probs - batch["log_probs"])
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(
            ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon
        )
        policy_loss = -torch.min(policy_loss_1, policy_loss_2 * advantages).mean()

        # Value loss.
        value_loss = F.mse_loss(values.squeeze(), batch["returns"])

        total_loss = (
            policy_loss
            + self.config.value_coeff * value_loss
            - self.config.entropy_coeff * entropy_bonus
        )

        # self.log("train/value_loss", value_loss, prog_bar=False, logger=True)
        # self.log("train/policy_loss", policy_loss, prog_bar=False, logger=True)
        # self.log("train/loss", total_loss, prog_bar=False, logger=True)
        self.log_dict(
            {
                "train/value_loss": value_loss,
                "train/policy_loss": policy_loss,
                "train/loss": total_loss,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=False,
        )

        return total_loss

    def collocate_data(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def training_step(self, batch, batch_idx):
        # Get the optimizer (if you have one)
        optimizer = self.optimizers()

        # Get a new data loader (e.g., for RL environment or custom sampling)
        data_loader = self.get_new_rollout()
        # Iterate over the data loader (or just use a single batch)
        for _ in range(self.config.update_epochs):
            for batch in data_loader:
                batch = self.collocate_data(batch)

                # Forward pass
                loss = self.actor_critic.loss(batch)

                # Manual optimization steps
                optimizer.zero_grad()  # Clear gradients
                self.manual_backward(loss)  # Backward pass

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.parameters(), max_norm=self.config.grad_clip
                )  # Clip gradients

                optimizer.step()  # Update weights

                # Logging(optional)
                self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Return None or a dictionary (optional)
        return None

    def configure_optimizers(self):
        return optim.Adam(self.actor_critic.parameters(), lr=self.config.lr)
