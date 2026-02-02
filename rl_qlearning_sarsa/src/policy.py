import numpy as np

def epsilon_greedy(Q: np.ndarray, state: int, epsilon: float, n_actions: int) -> int:
    """ε-greedy Aktionswahl."""
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[state]))
