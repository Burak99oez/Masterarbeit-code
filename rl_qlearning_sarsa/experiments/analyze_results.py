import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]   # .../rl_qlearning_sarsa
# results and figures live at the package root, not under experiments
RESULTS_DIR = BASE_DIR
FIGURES_DIR = BASE_DIR / "figures"
OUT_AGG_CSV = RESULTS_DIR / "aggregated_summary.csv"
SUMMARY_PATTERN = str(RESULTS_DIR / "*_summary.csv")

def load_summaries(pattern: str = SUMMARY_PATTERN) -> pd.DataFrame:
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"Keine Summary-Dateien gefunden unter: {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # jede summary ist bei dir 1 Zeile + Header; wir hängen sie zusammen
        dfs.append(df)
    all_df = pd.concat(dfs, ignore_index=True)

    # Datentypen sicher setzen
    num_cols = [
        "seed", "alpha", "gamma", "eps_start", "eps_end", "eps_decay",
        "eval_mean_return", "eval_std_return", "eval_mean_length", "eval_success_rate"
    ]
    for c in num_cols:
        if c in all_df.columns:
            all_df[c] = pd.to_numeric(all_df[c], errors="coerce")

    return all_df


def aggregate_over_seeds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregiert über Seeds: Mean/Std von eval_* Metriken je (algorithm, env).
    """
    group_cols = ["algorithm", "env"]

    agg = df.groupby(group_cols).agg(
        n_seeds=("seed", "count"),
        mean_eval_return=("eval_mean_return", "mean"),
        std_eval_return=("eval_mean_return", "std"),
        mean_eval_length=("eval_mean_length", "mean"),
        std_eval_length=("eval_mean_length", "std"),
        mean_success=("eval_success_rate", "mean"),
        std_success=("eval_success_rate", "std"),
    ).reset_index()

    # Schönere Rundung für Export (optional)
    for c in agg.columns:
        if c.startswith("mean_") or c.startswith("std_"):
            agg[c] = agg[c].round(4)

    return agg


def save_aggregate_table(agg: pd.DataFrame, out_csv: str = OUT_AGG_CSV) -> None:
    agg.to_csv(out_csv, index=False)


def plot_bar_metric(agg: pd.DataFrame, env: str, metric_mean: str, metric_std: str, title: str, filename: str):
    """
    Balkenplot pro Algorithmus mit Fehlerbalken (Std über Seeds).
    """
    os.makedirs(FIGURES_DIR, exist_ok=True)

    sub = agg[agg["env"] == env].copy()
    sub = sub.sort_values("algorithm")

    x = list(sub["algorithm"])
    y = list(sub[metric_mean])
    yerr = list(sub[metric_std].fillna(0.0))

    plt.figure()
    plt.bar(x, y, yerr=yerr, capsize=6)
    plt.title(title)
    plt.ylabel(metric_mean)
    plt.xlabel("algorithm")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=200)
    plt.close()


def create_plots(agg: pd.DataFrame) -> None:
    """
    Erstellt Standardplots für FrozenLake und Taxi.
    """
    envs = sorted(agg["env"].unique())

    for env in envs:
        # Success Rate
        plot_bar_metric(
            agg, env,
            metric_mean="mean_success",
            metric_std="std_success",
            title=f"Erfolgsrate (Mean ± Std über Seeds) – {env}",
            filename=f"{env}_success_rate.png".replace("/", "_")
        )

        # Mean Return
        plot_bar_metric(
            agg, env,
            metric_mean="mean_eval_return",
            metric_std="std_eval_return",
            title=f"Evaluation Return (Mean ± Std über Seeds) – {env}",
            filename=f"{env}_eval_return.png".replace("/", "_")
        )

        # Mean Episode Length
        plot_bar_metric(
            agg, env,
            metric_mean="mean_eval_length",
            metric_std="std_eval_length",
            title=f"Episodenlänge (Mean ± Std über Seeds) – {env}",
            filename=f"{env}_episode_length.png".replace("/", "_")
        )


def print_table_for_thesis(agg: pd.DataFrame) -> None:
    """
    Druckt eine kompakte Tabelle (Mean ± Std) in Textform,
    die du direkt in Kapitel 10 übernehmen kannst.
    """
    lines = []
    for _, row in agg.iterrows():
        lines.append(
            f"{row['env']} | {row['algorithm']} | "
            f"Success: {row['mean_success']} ± {row['std_success']} | "
            f"Return: {row['mean_eval_return']} ± {row['std_eval_return']} | "
            f"Len: {row['mean_eval_length']} ± {row['std_eval_length']} | "
            f"n={int(row['n_seeds'])}"
        )
    print("\n".join(lines))


def main():
    df = load_summaries()
    agg = aggregate_over_seeds(df)
    save_aggregate_table(agg)
    create_plots(agg)

    print("✅ Aggregierte Tabelle gespeichert:", OUT_AGG_CSV)
    print("✅ Plots gespeichert in:", FIGURES_DIR)
    print("\n--- Tabelle (für Kapitel 10) ---")
    print_table_for_thesis(agg)


if __name__ == "__main__":
    main()
