"""
This script processes the ablation results CSV files to note which is the best performing method in each scenario, and which methods are statistically overlapping with it.
"""

import pandas as pd
import numpy as np
import glob
import os
from argparse import ArgumentParser
from pathlib import Path

# --- Configuration ---

DEFAULT_BASE_PATH = Path(__file__).resolve().parents[1] / "data" / "ablations"

scenarios_mdp = [
    ("Mountain Car MDP", "ProbMountainCarMDPSimple/data"),
    ("Hill Car MDP", "ProbMountainCarMDPODE/data"),
    ("Lunar Lander MDP", "LunarLanderMDP/data"),
]

scenarios_pomdp = [
    ("Mountain Car POMDP", "ProbMountainCarPOMDPSimple/data"),
    ("Hill Car POMDP", "ProbMountainCarPOMDPODE/data"),
    ("Lunar Lander POMDP", "LunarLanderPOMDP/data"),
    ("2D Light-Dark", "CLD_2_1_psigma_0.1/data"),
    ("3D Light-Dark", "CLD_3_1_psigma_0.1/data"),
    ("4D Light-Dark", "CLD_4_1_psigma_0.1/data"),
    ("2-2D Light-Dark", "CLD_2_2_psigma_0.1/data"),
]

# Order matters for "Budget Level 0, 1, 2..." if we want to label them, 
# but we will primarily use the index 0..N-1 to compare.

methods_mdp = ["DPW", "AG-DPW", "VPW", "AG-VPW"]
methods_pomdp = ["PFT-DPW", "POMCPOW", "AG-PFT-DPW", "PFT-VPW", "AG-PFT-VPW", "VOMCPOW"]

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

# Helper for pretty names
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

def load_scenario_data(base_path, scenario_rel_path, method_list):
    full_path = os.path.join(base_path, scenario_rel_path)
    csv_files = glob.glob(os.path.join(full_path, "*.csv"))
    
    if not csv_files:
        print(f"Warning: No CSV files found in {full_path}")
        return {}

    # Load all CSVs
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not dfs:
        return {}
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["solver_name"] = full_df["solver_name"].replace(legacy_solver_names)
    
    # Structure: Dictionary mapping method_name -> sorted list of (max_query, mean, sem)
    data_by_method = {}
    
    for method in method_list:
        # Filter by solver_name
        m_df = full_df[full_df["solver_name"] == method]
        if m_df.empty:
            continue
        
        # Check if 'max_query' exists
        if 'max_query' not in m_df.columns:
            print(f"Warning: 'max_query' column missing for {method} in {scenario_rel_path}")
            continue
            
        m_df_sorted = m_df.sort_values(by="max_query")
        
        # Extract relevant columns
        data = []
        for _, row in m_df_sorted.iterrows():
            data.append({
                "query": int(row["max_query"]),
                "mean": float(row["reward_mean"]),
                "sem": float(row["reward_se"])
            })
        
        data_by_method[method] = data
        
    return data_by_method

def format_val(mean, sem):
    return f"{mean:.2f} ± {sem:.2f}"

def process_scenarios(base_path, scenarios, methods):
    for name, path in scenarios:
        print(f"\n=== Scenario: {name} ===")
        data = load_scenario_data(base_path, path, methods)
        
        if not data:
            print("  No data available.")
            continue
            
        # Determine number of budget levels based on the first available method
        # Ideally all methods have the same number of levels.
        # We'll take the max length found.
        max_levels = 0
        reference_method = None
        for m, d in data.items():
            if len(d) > max_levels:
                max_levels = len(d)
                reference_method = m
                
        if max_levels == 0:
            print("  No valid data points found.")
            continue
            
        for i in range(max_levels):
            # Collect data for this level
            level_stats = []
            
            for method in methods:
                if method not in data:
                    continue
                method_data = data[method]
                if i < len(method_data):
                    entry = method_data[i]
                    level_stats.append({
                        "method": method,
                        "pretty_name": methods_names_pretty.get(method, method),
                        "mean": entry["mean"],
                        "sem": entry["sem"],
                        "query": entry["query"]
                    })
            
            if not level_stats:
                continue
                
            # Find best (max mean)
            best_entry = max(level_stats, key=lambda x: x["mean"])
            best_mean = best_entry["mean"]
            best_sem = best_entry["sem"]
            
            # Identify Bold and Daggers
            bolds = []
            daggers = []
            others = []
            
            for entry in level_stats:
                # Check for Bold (Top mean) - Using a small epsilon for float comparison if needed
                # But strictly it's the max.
                if entry == best_entry:
                    bolds.append(entry)
                else:
                    # Check overlap: |mu1 - mu2| <= 2*(sem1 + sem2)
                    diff = abs(entry["mean"] - best_mean)
                    threshold = 2 * (entry["sem"] + best_sem)
                    
                    if diff <= threshold:
                        daggers.append(entry)
                    else:
                        others.append(entry)
                        
            # Print for this level
            # Using the query count of the best method as the label, or generic index
            print(f"  [Budget Level {i+1} (approx {best_entry['query']} queries)]")
            
            # Helper to print entry
            def print_entry(e, tag=""):
                print(f"    {e['pretty_name']:<20}: {format_val(e['mean'], e['sem']):<20} (queries: {e['query']}) {tag}")

            for e in bolds:
                print_entry(e, "[BOLD]")
            for e in daggers:
                print_entry(e, "[DAGGER]")
            for e in others:
                print_entry(e, "")

if __name__ == "__main__":
    parser = ArgumentParser(description="Summarize ablation result tables.")
    parser.add_argument(
        "--base-path",
        default=str(DEFAULT_BASE_PATH),
        help="Directory containing scenario result folders with CSV files.",
    )
    args = parser.parse_args()

    print("Output Format: Algorithm : Mean ± SEM (Queries) [Tag]")
    print("  [BOLD]   = Best performing algorithm (Highest Mean)")
    print("  [DAGGER] = Overlapping with Best (within 2*SEM range)")
    print("-" * 60)
    print("Processing MDP Scenarios...")
    process_scenarios(args.base_path, scenarios_mdp, methods_mdp)
    
    print("\n" + "="*40 + "\n")
    
    print("Processing POMDP Scenarios...")
    process_scenarios(args.base_path, scenarios_pomdp, methods_pomdp)
