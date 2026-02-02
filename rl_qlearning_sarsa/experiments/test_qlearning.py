# ensure imports work when running this script directly
try:
    from src.runner import run_experiment
except Exception:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.append(str(repo_root))
    from src.runner import run_experiment

run_experiment(
    algorithm="qlearning",
    env_name="FrozenLake-v1",
    seed=0,
    episodes_train=50000,
    episodes_eval=50,
    alpha=0.2,
    gamma=0.99,
    eps_start=1.0,
    eps_end=0.5,
    eps_decay=0.9995,
    out_dir="results",
    is_slippery=False
)

print("Q-learning test finished")
