import numpy as np
import torch


class RolloutDataset:
    def __init__(
        self,
        obs,
        actions,
        rewards,
        next_obs,
        advantages,
        returns,
        log_probs=None,
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
        self.values = []
        self.next_values = []

    def add(
        self,
        obs,
        act,
        rew,
        next_obs,
        terminated,
        truncated,
        info,
        log_probs=None,
        values=None,
        next_values=None,
    ):
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
        self.values.append(values)
        self.next_values.append(next_values)

        self.adv = []

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_observations.clear()
        self.terminated.clear()
        self.truncated.clear()
        self.infos.clear()
        self.log_probs.clear()
        self.values.clear()
        self.next_values.clear()

    def compute_advantages_and_returns(
        self, gamma: float = 0.99, gae_lambda: float = 1.0
    ):
        """
        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        Args:
            gamma (float): Discount factor.
            gae_lambda (float): GAE lambda parameter.

        Returns:
            advantages (list): Advantage estimates for each timestep.
            returns (list): Computed returns (advantages + value estimates) for each timestep.
        """
        advantages = [0] * len(self.rewards)
        gae = 0

        # Iterate in reverse order over the collected experiences
        for step in reversed(range(len(self.rewards))):

            # If the episode ended at this timestep, treat it as terminal
            done = self.terminated[step] or self.truncated[step]
            next_value = 0 if done else self.next_values[step]
            delta = self.rewards[step] + gamma * next_value - self.values[step]

            # When done, the future advantage is zero
            gae = delta + gamma * gae_lambda * (0 if done else gae)
            advantages[step] = gae

        # Compute returns (target values) as advantages plus the baseline values
        returns = [adv + val for adv, val in zip(advantages, self.values)]
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

        advantages, returns = self.compute_advantages_and_returns(gamma, gae_lambda)

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
