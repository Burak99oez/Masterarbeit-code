import numpy as np
from .policy import epsilon_greedy

def train_sarsa(env, episodes: int, alpha: float, gamma: float,
                eps_start: float, eps_end: float, eps_decay: float,
                seed: int):
    """
    Trainiert SARSA tabellarisch.
    Returns: Q, episode_returns, episode_lengths, episode_success
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    returns, lengths, success = [], [], []

    epsilon = eps_start

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        a = epsilon_greedy(Q, obs, epsilon, n_actions)

        done = False
        ep_return = 0.0
        steps = 0
        reached_goal = 0

        while not done:
            next_obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            if done:
                td_target = r  # terminal
                a_next = 0
            else:
                a_next = epsilon_greedy(Q, next_obs, epsilon, n_actions)
                td_target = r + gamma * Q[next_obs, a_next]

            td_error = td_target - Q[obs, a]
            Q[obs, a] += alpha * td_error

            obs, a = next_obs, a_next
            ep_return += r
            steps += 1

            if terminated and r > 0:
                reached_goal = 1

        returns.append(ep_return)
        lengths.append(steps)
        success.append(reached_goal)

        epsilon = max(eps_end, epsilon * eps_decay)

    return Q, np.array(returns), np.array(lengths), np.array(success)
