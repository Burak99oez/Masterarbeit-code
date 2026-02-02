import numpy as np
from .policy import epsilon_greedy

def train_qlearning(env, episodes: int, alpha: float, gamma: float,
                    eps_start: float, eps_end: float, eps_decay: float,
                    seed: int):
    """
    Trainiert Q-Learning tabellarisch.
    Returns: Q, episode_returns, episode_lengths, episode_success
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    returns, lengths, success = [], [], []

    epsilon = eps_start

    for ep in range(episodes):
        # reset that supports both Gymnasium and older Gym
        res = env.reset(seed=seed + ep)
        try:
            obs, _ = res
        except Exception:
            obs = res

        done = False
        ep_return = 0.0
        steps = 0
        reached_goal = 0

        while not done:
            a = epsilon_greedy(Q, obs, epsilon, n_actions)
            result = env.step(a)
            # support Gymnasium (obs, reward, terminated, truncated, info)
            # and older Gym (obs, reward, done, info)
            if isinstance(result, tuple) and len(result) == 5:
                next_obs, r, terminated, truncated, _ = result
            else:
                next_obs, r, done_flag, _ = result
                terminated = done_flag
                truncated = False
            done = terminated or truncated

            # Q-Learning Update
            td_target = r + gamma * np.max(Q[next_obs])
            td_error = td_target - Q[obs, a]
            Q[obs, a] += alpha * td_error

            obs = next_obs
            ep_return += r
            steps += 1

            # FrozenLake: Erfolg wenn terminated und reward==1
            if terminated and r > 0:
                reached_goal = 1

        returns.append(ep_return)
        lengths.append(steps)
        success.append(reached_goal)

        # epsilon decay
        epsilon = max(eps_end, epsilon * eps_decay)

    return Q, np.array(returns), np.array(lengths), np.array(success)
