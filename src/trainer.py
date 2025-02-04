import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader

from actor_critic import ActorCritic
from buffer import RolloutBuffer


class PPOTrainer(pl.LightningModule):
    def __init__(
        self,
        actor_critic: ActorCritic,
        env,
        rollout_length: int,
        update_epochs: int,
        lr=3e-4,
        gamma=0.99,
        clip_epsilon=0.2,
        value_coeff=0.5,
        entropy_coeff=0.01,
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
        self.rollout_length = rollout_length
        self.update_epochs = update_epochs
        self.lr = lr
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff

        # Rollout buffer should have methods to add transitions, get a dataset, and clear itself.
        self.buffer = RolloutBuffer()
        self.current_update_epoch = 0  # counts epochs since last rollout collection
        self.rollout_dataset = None

    def forward(self, x):
        return self.actor_critic(x)

    def collect_rollout(self):
        """Collects a rollout using the current policy."""
        state = self.env.reset()
        for _ in range(self.rollout_length):
            # Convert state to tensor and add batch dimension.
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_logits, _ = self.actor_critic(state_tensor)
            dist = Categorical(logits=action_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step in the environment
            next_state, reward, terminated, truncated, info = self.env.step(
                action.item()
            )

            # Store the transition (assuming your buffer.add signature matches these arguments)
            self.buffer.add(
                obs=state,
                act=action.item(),
                rew=reward,
                next_obs=next_state,
                terminated=terminated,
                truncated=truncated,
                info=info,
                log_probs=log_prob.item(),
            )

            state = next_state
            if terminated or truncated:
                state = self.env.reset()

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
        if self.current_update_epoch >= self.update_epochs:
            self.current_update_epoch = 0

        return DataLoader(self.rollout_dataset, batch_size=64, shuffle=True)

    def compute_loss(self, batch):
        """
        Unpacks the batch and computes the PPO loss.
        Expects the batch to provide:
          states, actions, old_log_probs, returns, advantages.
        """
        states, actions, old_log_probs, returns, advantages = batch

        # Forward pass through actor-critic.
        action_logits, values = self.actor_critic(states)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)

        # Compute probability ratio.
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

        # Policy loss.
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # Value loss.
        value_loss = nn.functional.mse_loss(values.squeeze(), returns)

        # Entropy bonus.
        entropy_bonus = dist.entropy().mean()

        total_loss = (
            policy_loss
            + self.value_coeff * value_loss
            - self.entropy_coeff * entropy_bonus
        )
        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self.compute_loss(batch)
        self.log("loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.actor_critic.parameters(), lr=self.lr)
