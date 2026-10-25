# H1 bridge: Task 2 test predictions -> H1 real mode

## What was done
- Loaded the existing `train.csv.gz` and `test.csv.gz`.
- Fit LogisticRegression and HistGradientBoosting using the preprocessing/model settings in `experiments_clean-3(2).ipynb`.
- Estimated the transition baseline using train only.
- Produced `h1_real_input.csv.gz` with columns `y`, `p_baseline`, `p_logistic`, `p_hist_gradient_boosting`, plus `idind` and `target_year`.
- Ran the H1 real-mode command on this input and saved outputs in `real_pilot/`.

## Important interpretation
The test set has fully observed labels (`hidden=0`). Therefore this is a first integration/smoke run, not an experiment demonstrating H1 under missing outcomes. C0/C1 bounds collapse to exact pairwise differences when all labels are observed, and the true regret fields are intentionally left NaN by H1's real mode. C2 outputs are scenario sensitivity, not unconditional guarantees.

The observed test Brier scores are:
- transition baseline: 0.063989
- logistic regression: 0.060968
- HistGradientBoosting: 0.057416

These are descriptive scores on the supplied test split. Do not use the test set to tune models or choose H1 parameters.

## Files
- `build_h1_real_input.py`: reproducible bridge script.
- `h1_real_input.csv.gz`: generated H1 input.
- `real_pilot/`: outputs from H1 real mode.
- `h1_pilot_summary.md`: summary of this run.

## Reproduce
From the H1 project root (the folder containing `src/` and `configs/`):
```bash
python -m src.experiments real --input /path/to/h1_real_input.csv.gz --out results/real_pilot --config configs/h1_default.json
```
To regenerate the input table, use the bridge script and point `--data-dir` to the directory containing `train.csv.gz` and `test.csv.gz`.
