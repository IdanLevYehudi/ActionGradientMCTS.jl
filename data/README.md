# Data Layout

The analysis scripts expect this layout by default:

```text
data/ablations/
  ProbMountainCarMDPSimple/data/*.csv
  ProbMountainCarMDPODE/data/*.csv
  LunarLanderMDP/data/*.csv
  ProbMountainCarPOMDPSimple/data/*.csv
  ProbMountainCarPOMDPODE/data/*.csv
  LunarLanderPOMDP/data/*.csv
  CLD_2_1_psigma_0.1/data/*.csv
  CLD_3_1_psigma_0.1/data/*.csv
  CLD_4_1_psigma_0.1/data/*.csv
  CLD_2_2_psigma_0.1/data/*.csv
```

Pass `--base-path /path/to/results` to `analysis/process_tables.py` or `analysis/ablation_trend_plots.py` to use a different local result directory.
