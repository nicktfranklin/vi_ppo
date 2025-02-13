import matplotlib.pyplot as plt
import numpy as np

from vi_ppo.buffer import RolloutBuffer

# from thread_the_needle.gridworld import GridWorld


def visualize_rollout(env, buffer: RolloutBuffer, max_steps: int | None = None):
    """
    Visualize the rollout in the environment.

    Args:
        max_steps (int): Maximum number of steps to visualize.
    """
    max_steps = min(len(buffer), max_steps) if max_steps is not None else len(buffer)

    states = [None] * max_steps
    next_states = [None] * max_steps

    for ii in range(max_steps):
        states[ii] = buffer.infos[ii]["start_state"]
        next_states[ii] = buffer.infos[ii]["successor_state"]

    trajectories = [[]]
    for ii in range(len(states)):
        if buffer.terminated[ii] or buffer.truncated[ii]:
            trajectories.append([])
        elif ii > 0 and states[ii] != next_states[ii - 1]:
            trajectories.append([])
        x, y = env.observation_model.get_grid_coords(states[ii])
        x += np.random.randn() * 0.1
        y += np.random.randn() * 0.1
        trajectories[-1].append((x, y))

    plt.figure(figsize=(8, 8))
    for trajectory in trajectories:
        if len(trajectory) > 0:
            x, y = zip(*trajectory)
            plt.plot(y, x, marker="o")

    # plt.figure()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Rollout Visualization")
    plt.xlim(0, 20)
    plt.ylim(20, 0)
    plt.grid(True)
    plt.show()

    plt.figure()
    obs = np.stack(buffer.observations).mean(axis=0).squeeze()
    plt.imshow(obs, cmap="gray")
    plt.title("Average observation in rollout")
    plt.show()
