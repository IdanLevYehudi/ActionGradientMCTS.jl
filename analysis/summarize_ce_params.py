"""
This script is used to summarize the best parameters from the output of a CE optimization run.
It scans a specified directory for CSV files named 'ce_params*.csv', identifies the row with
the maximum 'reward_mean' in each file, and outputs a summary to stdout containing the
filename, row index, reward_mean, reward_se, and other relevant parameters.
"""

#!/usr/bin/env python3
import argparse
import csv
import sys
from pathlib import Path


def find_max_reward_row(csv_path):
    """
    Returns (row_index, reward_mean, reward_se) for the row with max reward_mean.
    row_index counts data rows starting from 1 (header row not counted).
    """
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        best_row = None
        best_reward = None
        best_index = None

        # Enumerate data rows starting at 1 (since header is not counted)
        for i, row in enumerate(reader, start=1):
            try:
                reward_mean = float(row["reward_mean"])
            except (KeyError, ValueError):
                # If column missing or not a float, skip this row
                continue

            if best_reward is None or reward_mean > best_reward:
                best_reward = reward_mean
                best_row = row
                best_index = i

        if best_row is None:
            return None

        # Safely parse reward_se (if missing or invalid, just leave as empty string)
        keys_to_print = ["ucb_c", "k_act", "alpha_act", "k_obs", "alpha_obs", "lr", "reward_se"]
        reward_se = None
        ret_vals = []
        for key in keys_to_print:
            if key in best_row:
                csv_val = best_row[key]
                try:
                    csv_val = float(csv_val)
                    if key == "reward_se":
                        reward_se = csv_val
                    else:
                        ret_vals.append(csv_val)
                except (TypeError, ValueError):
                    pass

        return best_index, best_reward, reward_se, ret_vals


def main():
    parser = argparse.ArgumentParser(
        description="Summarize max reward_mean from ce_params*.csv files in a directory."
    )
    parser.add_argument(
        "directory",
        type=str,
        help="Path to directory containing ce_params*.csv files",
    )
    args = parser.parse_args()

    dir_path = Path(args.directory)
    if not dir_path.is_dir():
        print(f"Error: {dir_path} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Prepare CSV writer to stdout
    writer = csv.writer(sys.stdout)
    writer.writerow(["filename", "row_index", "reward_mean", "reward_se"])

    files_found = False
    for csv_path in sorted(dir_path.glob("ce_params*.csv")):
        files_found = True
        result = find_max_reward_row(csv_path)
        if result is None:
            # No valid reward_mean row found
            print(f"Warning: no valid reward_mean found in {csv_path.name}", file=sys.stderr)
            continue

        row_index, reward_mean, reward_se, ret_vals = result
        writer.writerow([csv_path.name, row_index, reward_mean, reward_se, ret_vals])

    if not files_found:
        print(f"No files matching 'ce_params*.csv' found in {dir_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
