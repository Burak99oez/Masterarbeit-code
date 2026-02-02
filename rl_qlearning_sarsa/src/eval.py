import numpy as np

def evaluate_greedy(env, Q: np.ndarray, episodes: int, seed: int):
    """Evaluation ohne Exploration (greedy)."""
    returns, lengths, success = [], [], []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + 10_000 + ep)
        done = False
        ep_return = 0.0
        steps = 0
        reached_goal = 0

        while not done:
            a = int(np.argmax(Q[obs]))
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_return += r
            steps += 1
            if terminated and r > 0:
                reached_goal = 1

        returns.append(ep_return)
        lengths.append(steps)
        success.append(reached_goal)

    return np.mean(returns), np.std(returns), np.mean(lengths), np.mean(success)
