import csv
import gymnasium as gym
from .utils import ensure_dir, set_seed
from .qlearning import train_qlearning
from .sarsa import train_sarsa
from .eval import evaluate_greedy

def make_env(env_name: str, is_slippery: bool = True):
    if env_name.lower().startswith("frozenlake"):
        return gym.make("FrozenLake-v1", is_slippery=is_slippery)
    elif env_name.lower().startswith("taxi"):
        return gym.make("Taxi-v3")
    else:
        raise ValueError(f"Unknown env: {env_name}")

def run_experiment(algorithm: str, env_name: str, seed: int,
                   episodes_train: int, episodes_eval: int,
                   alpha: float, gamma: float,
                   eps_start: float, eps_end: float, eps_decay: float,
                   out_dir: str, is_slippery: bool = True):

    ensure_dir(out_dir)
    set_seed(seed)

    env = make_env(env_name, is_slippery=is_slippery)

    if algorithm == "qlearning":
        Q, returns, lengths, success = train_qlearning(
            env, episodes_train, alpha, gamma, eps_start, eps_end, eps_decay, seed
        )
    elif algorithm == "sarsa":
        Q, returns, lengths, success = train_sarsa(
            env, episodes_train, alpha, gamma, eps_start, eps_end, eps_decay, seed
        )
    else:
        raise ValueError("algorithm must be 'qlearning' or 'sarsa'")

    mean_r, std_r, mean_len, mean_succ = evaluate_greedy(env, Q, episodes_eval, seed)

    # Save training log
    filename = f"{algorithm}_{env_name}_seed{seed}_train.csv".replace("/", "_")
    with open(f"{out_dir}/{filename}", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return", "length", "success"])
        for i in range(len(returns)):
            w.writerow([i, float(returns[i]), int(lengths[i]), int(success[i])])

    # Save summary
    summary_file = f"{out_dir}/{algorithm}_{env_name}_seed{seed}_summary.csv".replace("/", "_")
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algorithm", "env", "seed", "alpha", "gamma", "eps_start", "eps_end", "eps_decay",
                    "eval_mean_return", "eval_std_return", "eval_mean_length", "eval_success_rate"])
        w.writerow([algorithm, env_name, seed, alpha, gamma, eps_start, eps_end, eps_decay,
                    mean_r, std_r, mean_len, mean_succ])

    env.close()

    return {
        "eval_mean_return": mean_r,
        "eval_std_return": std_r,
        "eval_mean_length": mean_len,
        "eval_success_rate": mean_succ,
    }
