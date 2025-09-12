from pathlib import Path

import fire

from repro.data_loading import (
    DEFAULT_PLOTS_PATH,
    DEFAULT_RESULTS_PATH,
    get_sweep_df,
    load_sweep_data,
)
from repro.sec7_3_rl_train.plot.plot_shared import (
    FRAMEWORKS_ORDERED,
    create_generic_metrics_panel,
)
from repro.sec7_3_rl_train.run_small_to_med_scale import (
    PPO_SMALL_TO_MED_SWEEPS,
    PPO_SWEEP_PARAM_BASE,
    SMALL_TO_MED_EXPERIMENT_BASE_NAME,
)
from repro.sec7_3_rl_train.shared import RL_TRAIN_DIR, SYS, get_experiment_name_and_results_path


def plot_small_to_med_scale(
    results_path: str = DEFAULT_RESULTS_PATH, plots_path: str = DEFAULT_PLOTS_PATH
):
    """
    Generate plots for small to medium scale RL training experiments.

    Args:
        results_path (str): Path to the results directory. Defaults to "./results".
        plots_path (str): Path to the plots directory. Defaults to "./plots".
    """

    base_path = str(Path(results_path) / RL_TRAIN_DIR / SMALL_TO_MED_EXPERIMENT_BASE_NAME)

    print("Loading data...")
    data = load_sweep_data(
        base_path,
        PPO_SWEEP_PARAM_BASE,
        PPO_SMALL_TO_MED_SWEEPS,
        name_function=get_experiment_name_and_results_path,
        systems=SYS,
    )

    print("Building dataframes...")
    df_ep_len = get_sweep_df(data, PPO_SMALL_TO_MED_SWEEPS, "ep_len", SYS)
    df_params = get_sweep_df(data, PPO_SMALL_TO_MED_SWEEPS, "params_per_layer", SYS)
    df_layers = get_sweep_df(data, PPO_SMALL_TO_MED_SWEEPS, "num_layers", SYS)
    df_layers_saved = df_layers.copy()
    df_envs = get_sweep_df(data, PPO_SMALL_TO_MED_SWEEPS, "num_envs", SYS)

    sweep_values_ordered = [
        PPO_SMALL_TO_MED_SWEEPS["ep_len"],
        PPO_SMALL_TO_MED_SWEEPS["params_per_layer"],
        PPO_SMALL_TO_MED_SWEEPS["num_layers"],
        PPO_SMALL_TO_MED_SWEEPS["num_envs"],
    ]
    sweep_keys = ["ep_len", "params_per_layer", "num_layers", "num_envs"]

    framework_order = FRAMEWORKS_ORDERED.copy()
    framework_order.remove("cleanrlcache")

    print("Generating panel plot...")
    fig = create_generic_metrics_panel(
        sweep_dfs=[df_ep_len, df_params, df_layers, df_envs],
        frameworks_ordered=framework_order,
        metrics=["iter_mean", "gpu_util"],
        sweep_keys=sweep_keys,
        sweep_values_ordered=sweep_values_ordered,
        figsize=(30, 7),
        y_lim_iter_time=(pow(10, -1), pow(10, 1.4)),
        bbox_to_anchor=(0.5, 1.05),
        text_size=24,
    )

    print("Saving plot...")
    # Create plots directory if it doesn't exist
    plot_dir = Path(plots_path) / RL_TRAIN_DIR / SMALL_TO_MED_EXPERIMENT_BASE_NAME
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = plot_dir / "small_to_med_scale.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {out_pdf}")

    print("========== cleanrl vs cleanrlcache ===========")
    df_layers_filtered = df_layers_saved[
        df_layers_saved["framework"].isin(["cleanrl", "cleanrlcache"])
    ]
    # NOTE: renormalize iter_mean
    for nl in PPO_SMALL_TO_MED_SWEEPS["num_layers"]:
        df_layers_filtered_nl = df_layers_filtered[df_layers_filtered["sweep_value"] == nl]
        df_layers_filtered_nl["iter_mean_ratio"] = (
            df_layers_filtered_nl["iter_mean_ratio"]
            / df_layers_filtered_nl["iter_mean_ratio"].min()
        )
        df_layers_filtered.loc[df_layers_filtered["sweep_value"] == nl, "iter_mean_ratio"] = (
            df_layers_filtered_nl["iter_mean_ratio"]
        )
    fig = create_generic_metrics_panel(
        sweep_dfs=[df_layers_filtered],
        frameworks_ordered=["cleanrl", "cleanrlcache"],
        metrics=["iter_mean"],
        sweep_keys=["num_layers"],
        sweep_values_ordered=[PPO_SMALL_TO_MED_SWEEPS["num_layers"]],
        figsize=(10, 2),
        bbox_to_anchor=(0.5, 0.92),
        text_size=18,
    )

    print("Saving plot...")
    # Create plots directory if it doesn't exist
    plot_dir = Path(plots_path) / RL_TRAIN_DIR / SMALL_TO_MED_EXPERIMENT_BASE_NAME
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = plot_dir / "cleanrl_vs_cleanrlcache.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Plot saved as {out_pdf}")


if __name__ == "__main__":
    fire.Fire(plot_small_to_med_scale)
