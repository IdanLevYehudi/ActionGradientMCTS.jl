"""
This script generates ablation trend plots for MDP and POMDP scenarios based on experimental ablation results.
"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import glob
import os
from argparse import ArgumentParser
from pathlib import Path

DEFAULT_BASE_PATH = Path(__file__).resolve().parents[1] / "data" / "ablations"

scenarios_mdp = [
    "Mountain Car MDP",
    "Hill Car MDP",
    "Lunar Lander MDP",
]

scenarios_pomdp = [
    "Mountain Car POMDP",
    "Hill Car POMDP",
    "Lunar Lander POMDP",
    "2D Light-Dark",
    "3D Light-Dark",
    "4D Light-Dark",
    "2-2D Light-Dark",
]

scenarios_plotting_order = [
    (scenarios_mdp, 0),
    (scenarios_mdp, 1),
    (scenarios_mdp, 2),
    (scenarios_pomdp, 3),
    (scenarios_pomdp, 4),
    (scenarios_pomdp, 0),
    (scenarios_pomdp, 1),
    (scenarios_pomdp, 2),
    (scenarios_pomdp, 5),
    (scenarios_pomdp, 6),
]

paths_mdps = [
    "ProbMountainCarMDPSimple/data",  # MountainCarMDP
    "ProbMountainCarMDPODE/data",  # HillCarMDP
    "LunarLanderMDP/data",  # LunarLanderMDP
]

paths_pomdps = [
    "ProbMountainCarPOMDPSimple/data",  # MountainCarPOMDP
    "ProbMountainCarPOMDPODE/data",  # "HillCarPOMDP",
    "LunarLanderPOMDP/data",  # "LunarLanderPOMDP",
    "CLD_2_1_psigma_0.1/data",  # "LightDark2D",
    "CLD_3_1_psigma_0.1/data",  # "LightDark3D",
    "CLD_4_1_psigma_0.1/data",  # "LightDark4D",
    "CLD_2_2_psigma_0.1/data",  # "LightDark2Dx2",
]

budgets_mdp = np.array(
    [
        [50, 89, 158, 281, 500],
        [50, 89, 158, 281, 500],
        [100, 178, 316, 562, 1000],
    ]
)

budgets_pomdp = np.array(
    [
        [50, 89, 158, 281, 500],
        [50, 89, 158, 281, 500],
        [100, 178, 316, 562, 1000],
        [50, 89, 158, 281, 500],
        [50, 89, 158, 281, 500],
        [50, 89, 158, 281, 500],
        [50, 89, 158, 281, 500],
    ]
)


methods_raw_mdp = ["DPW", "AG-DPW", "VPW", "AG-VPW"]
methods_raw_pomdp = ["PFT-DPW", "POMCPOW", "AG-PFT-DPW", "PFT-VPW", "AG-PFT-VPW", "VOMCPOW"]
methods_raw = methods_raw_mdp + methods_raw_pomdp

legacy_solver_names = {
    "PFT-DPW-MDP": "DPW",
    "AG-DPW-MDP": "AG-DPW",
    "PFT-VPW-MDP": "VPW",
    "AG-VPW-MDP": "AG-VPW",
    "PFT-DPW-POMDP": "PFT-DPW",
    "AG-DPW-POMDP": "AG-PFT-DPW",
    "PFT-VPW-POMDP": "PFT-VPW",
    "AG-VPW-POMDP": "AG-PFT-VPW",
}

methods_names_pretty = {
    "DPW": "DPW",
    "AG-DPW": "AG-DPW (Ours)",
    "VPW": "VPW",
    "AG-VPW": "AG-VPW (Ours)",
    "PFT-DPW": "PFT-DPW",
    "POMCPOW": "POMCPOW",
    "AG-PFT-DPW": "AG-PFT-DPW (Ours)",
    "PFT-VPW": "PFT-VPW",
    "AG-PFT-VPW": "AG-PFT-VPW (Ours)",
    "VOMCPOW": "VOMCPOW"
}
method_colors = {
    "DPW": "tab:orange",
    "AG-DPW": "tab:blue",
    "VPW": "tab:orange",
    "AG-VPW": "tab:blue",
    "PFT-DPW": "tab:orange",
    "POMCPOW": "tab:green",
    "AG-PFT-DPW": "tab:blue",
    "PFT-VPW": "tab:orange",
    "AG-PFT-VPW": "tab:blue",
    "VOMCPOW": "tab:green",
}

# Full (solid) lines for DPW / POMCPOW methods, dashed for VPW / VOMCPOW methods
method_linestyle = {
    "DPW": "-",
    "AG-DPW": "-",
    "VPW": "--",
    "AG-VPW": "--",
    "PFT-DPW": "-",
    "POMCPOW": "-",
    "AG-PFT-DPW": "-",
    "PFT-VPW": "--",
    "AG-PFT-VPW": "--",
    "VOMCPOW": "--",
}

methods_names_pretty_colors = {methods_names_pretty[k]: v for k, v in method_colors.items()}
methods_names_pretty_linestyle = {methods_names_pretty[k]: v for k, v in method_linestyle.items()}


def create_datastructure_results(scenario_dir, methods, budget_col):
    csv_files = glob.glob(os.path.join(scenario_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in directory: {scenario_dir}")

    df_list = [pd.read_csv(file) for file in csv_files]
    for df in df_list:
        df["solver_name"] = df["solver_name"].replace(legacy_solver_names)
    new_df = pd.DataFrame({"sim_budget": budget_col})
    for df in df_list:
        ### Extract the rows that have column "solver_name" value the same as the method name
        # Take these rows and assign them to new_df[name]
        for name in methods:
            method_rows = df[df["solver_name"] == name]
            if not method_rows.empty:
                # Assuming the column "normalized_score" contains the scores
                method_means = method_rows["reward_mean"].values
                method_ses = method_rows["reward_se"].values
                new_df[methods_names_pretty[name] + "-mean"] = method_means
                new_df[methods_names_pretty[name] + "-se"] = method_ses
    return new_df

fig, axes = plt.subplots(2, 5, figsize=(8.0, 3.0), dpi=300)

def load_results(base_path):
    mdp_dfs = [
        create_datastructure_results(os.path.join(base_path, path), methods_raw_mdp, budgets_mdp[i])
        for i, path in enumerate(paths_mdps)
    ]
    pomdp_dfs = [
        create_datastructure_results(os.path.join(base_path, path), methods_raw_pomdp, budgets_pomdp[i])
        for i, path in enumerate(paths_pomdps)
    ]
    return mdp_dfs, pomdp_dfs


def plot_results(ax, scenario_name, method_names, df):
    # for ax, scenario in zip(axes.flat, scenarios):
    for m in method_names:
        c = methods_names_pretty_colors[m]
        l = methods_names_pretty_linestyle[m]
        method_mean = df[m + "-mean"].values
        method_se = df[m + "-se"].values
        budgets = df["sim_budget"].values
        ax.plot(budgets, method_mean, label=m, color=c, lw=1.1, linestyle=l)
        ax.fill_between(
            budgets,
            method_mean - 2 * method_se,
            method_mean + 2 * method_se,
            color=c,
            alpha=0.25,
        )
    ax.set_xscale("log")  # Set x-axis to logarithmic scale
    ax.set_title(scenario_name, fontsize=7, pad=2)
    # ax.set_ylim(0, 100)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{int(b)}" for b in budgets], fontsize=6)  # Set tick labels
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.2, lw=0.5)
    # fig.text(0.5, 0.02, "Computation Budget (×10⁴ steps)", ha="center", fontsize=8)
    # fig.text(0.02, 0.5, "Avg. Normalized Score", va="center", rotation="vertical", fontsize=8)


def set_figure_labels(fig, method_names):
    handles, labels = fig.axes[0].get_legend_handles_labels()
    legend_handles = [
        plt.Line2D([0], [0], color=methods_names_pretty_colors[label], lw=1.5, linestyle=methods_names_pretty_linestyle[label])
        for label in method_names
    ]
    fig.legend(
        legend_handles,
        method_names,
        ncol=6,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        fontsize=7,
    )
    # fig.legend(method_names, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=7)
    fig.tight_layout(pad=0.1, h_pad=0.1, w_pad=0.1)


if __name__ == "__main__":
    parser = ArgumentParser(description="Plot ablation trends from CSV result folders.")
    parser.add_argument(
        "--base-path",
        default=str(DEFAULT_BASE_PATH),
        help="Directory containing scenario result folders with CSV files.",
    )
    parser.add_argument(
        "--output",
        default="mdp_pomdp_grid_results.pdf",
        help="Output path for the generated figure.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot interactively.")
    args = parser.parse_args()

    mdp_dfs, pomdp_dfs = load_results(args.base_path)
    # Plot MDP results
    ax_index = lambda i: (i // 5, i % 5)

    for i, (scenario_list, scenario_idx) in enumerate(scenarios_plotting_order):
        scenario = scenario_list[scenario_idx]
        mdp = scenario_list is scenarios_mdp
        method_names = [methods_names_pretty[m] for m in methods_raw_mdp] if mdp else [methods_names_pretty[m] for m in methods_raw_pomdp]
        df = mdp_dfs[scenario_idx] if mdp else pomdp_dfs[scenario_idx]
        plot_results(
            axes[ax_index(i)],
            scenario,
            method_names,
            df,
        )

    set_figure_labels(
        fig, ["AG-DPW (Ours)", "AG-VPW (Ours)", "DPW", "VPW", "POMCPOW", "VOMCPOW"]
    )

    fig.savefig(args.output, bbox_inches="tight")

    if args.show:
        plt.show()
