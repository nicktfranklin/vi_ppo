from collections import namedtuple

import numpy as np
import torch

# Define a simple dataset object (OARO tuple with GAE values).
# OARO stands for Observation, Action, Reward, Observation (successor observation)
OARODataset = namedtuple(
    "OARODataset",
    [
        "observations",
        "actions",
        "rewards",
        "next_observations",
        "advantages",
        "returns",
        "log_probs",
    ],
)


class RolloutDataset:
    def __init__(
        self, obs, actions, rewards, next_obs, advantages, returns, log_probs=None
    ):
        self.obs = self._stack(obs)
        self.actions = self._stack(actions)
        self.rewards = self._stack(rewards)
        self.next_obs = self._stack(next_obs)
        self.advantages = self._stack(advantages)
        self.returns = self._stack(returns)
        self.log_probs = self._stack(log_probs) if log_probs is not None else None

    @staticmethod
    def _stack(x: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(np.stack(x))
        if x.dtype == torch.int64 or x.dtype == torch.int32:
            return x.long()
        if x.dtype == torch.float64 or x.dtype == torch.float32:
            return x.float()
        raise ValueError(f"Unsupported dtype {x.dtype}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            "observations": self.obs[idx],
            "actions": self.actions[idx],
            "rewards": self.rewards[idx],
            "next_observations": self.next_obs[idx],
            "advantages": self.advantages[idx],
            "returns": self.returns[idx],
            "log_probs": self.log_probs[idx] if self.log_probs is not None else None,
        }


class RolloutBuffer:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []  # Successor observations
        self.terminated = []
        self.truncated = []
        self.infos = []
        self.log_probs = []

    def add(self, obs, act, rew, next_obs, terminated, truncated, info, log_probs=None):
        """
        Add a transition to the buffer.

        Args:
            obs: The current observation.
            act: The action taken.
            rew: The reward received.
            next_obs: The successor observation.
            terminated: Boolean flag indicating if the episode terminated.
            truncated: Boolean flag indicating if the episode was truncated.
            info: Additional info from the environment.
        """
        self.observations.append(obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.next_observations.append(next_obs)
        self.terminated.append(terminated)
        self.truncated.append(truncated)
        self.infos.append(info)
        self.log_probs.append(log_probs)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.infos.clear()
        self.log_probs.clear()

    def compute_gae(self, values, gamma: float = 0.99, gae_lambda: float = 1.0):
        """
        Compute Generalized Advantage Estimation (GAE) for the collected experiences.

        Args:
            values (list or np.array): Value estimates for each state in the buffer.
                                       It should have length len(self.rewards) + 1 (the extra element is
                                       the value estimate for the last next_observation).
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda parameter.

        Returns:
            advantages (list): Advantage estimates for each timestep.
            returns (list): Computed returns (advantages + value estimates) for each timestep.
        """
        assert (
            len(values) == len(self.rewards) + 1
        ), f"Expected len(values) == len(rewards) + 1, found lengths {len(values)} and {len(self.rewards)}"
        advantages = [0] * len(self.rewards)
        gae = 0
        # Iterate in reverse order over the collected experiences
        for t in reversed(range(len(self.rewards))):
            # If the episode ended at this timestep, treat it as terminal
            done = self.terminated[t] or self.truncated[t]
            next_value = 0 if done else values[t + 1]
            delta = self.rewards[t] + gamma * next_value - values[t]
            # When done, the future advantage is zero
            gae = delta + gamma * gae_lambda * (0 if done else gae)
            advantages[t] = gae
        # Compute returns (target values) as advantages plus the baseline values
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        return advantages, returns

    def compute_rewards_to_go(self, gamma):
        """
        Compute the Monte Carlo estimate of rewards-to-go for the collected experiences.

        Args:
            gamma (float): Discount factor.

        Returns:
            returns (list): Monte Carlo estimates of rewards-to-go for each timestep.
        """
        returns = [0] * (len(self.rewards) + 1)
        R = 0
        # Process the rewards in reverse order
        for t in reversed(range(len(self.rewards))):
            # Reset the cumulative reward if the episode ended at this step.
            if self.terminated[t] or self.truncated[t]:
                R = self.rewards[t]
            else:
                R = self.rewards[t] + gamma * R
            returns[t] = R
        return returns

    def get_dataset(
        self,
        values: list | np.ndarray | None = None,
        gamma: float = 0.99,
        gae_lambda: float = 1.0,
    ):
        """
        Compute the dataset containing the OARO tuple along with advantage and return estimates.

        If value estimates are provided, GAE is computed; otherwise, a Monte Carlo estimate of rewards-to-go is used
        for both returns and advantages.

        Args:
            gamma (float): Discount factor.
            lam (float): GAE lambda parameter.
            values (list or np.array, optional): Value estimates for each observation
                                                 (must have length len(rewards)+1). Defaults to None.

        Returns:
            OARODataset: A namedtuple with fields:
                - observations: list of observations.
                - actions: list of actions.
                - rewards: list of rewards.
                - next_observations: list of successor observations.
                - advantages: computed advantage estimates.
                - returns: computed return values.
        """

        if values is None:
            values = self.compute_rewards_to_go(gamma)
        advantages, returns = self.compute_gae(values, gamma, gae_lambda)

        # dataset = OARODataset(
        #     observations=torch.from_numpy(np.stack(self.observations)),
        #     actions=torch.from_numpy(np.stack(self.actions)),
        #     rewards=torch.from_numpy(np.stack(self.rewards)),
        #     next_observations=torch.from_numpy(np.stack(self.next_observations)),
        #     advantages=torch.from_numpy(np.stack(advantages)),
        #     returns=torch.from_numpy(np.stack(returns)),
        #     log_probs=(
        #         torch.from_numpy(np.stack(self.log_probs))
        #         if any(self.log_probs)
        #         else None
        #     ),
        # )

        dataset = RolloutDataset(
            obs=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            next_obs=self.next_observations,
            advantages=advantages,
            returns=returns,
            log_probs=self.log_probs,
        )
        return dataset
