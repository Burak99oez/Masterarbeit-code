# ensure imports work when running this script directly
try:
    from src.runner import run_experiment
except Exception:
    import sys
    from pathlib import Path
    # add the parent package folder (rl_qlearning_sarsa) to sys.path so `src` can be imported
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from src.runner import run_experiment

def main():
    seeds = [0, 1, 2, 3, 4]
    envs = ["FrozenLake-v1", "Taxi-v3"]
    algos = ["qlearning", "sarsa"]

    # Hyperparameter (Startwerte, später anpassen)
    alpha = 0.1
    gamma = 0.99
    eps_start = 1.0
    eps_end = 0.05
    eps_decay = 0.995

    episodes_train = 5000
    episodes_eval = 200

    for env in envs:
        for algo in algos:
            for seed in seeds:
                print(f"Running {algo} on {env} seed={seed}")
                run_experiment(
                    algorithm=algo,
                    env_name=env,
                    seed=seed,
                    episodes_train=episodes_train,
                    episodes_eval=episodes_eval,
                    alpha=alpha,
                    gamma=gamma,
                    eps_start=eps_start,
                    eps_end=eps_end,
                    eps_decay=eps_decay,
                    out_dir="results",
                    is_slippery=True
                )

if __name__ == "__main__":
    main()
