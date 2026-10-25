# H1 artificial-missingness pilot on Task 2 test predictions

## Run
- Input rows: 49,651 test observations.
- Candidate forecasts: transition baseline, logistic regression, HistGradientBoosting.
- Masking: independent row-level MCAR masks at 10%, 20%, 40%; 30 seeds per fraction.
- Methods: C0 and C1.
- Total output rows: 1,440 (2 methods x 3 missing fractions x 8 epsilon values x 30 seeds).

## Brier reference on the full labeled test split
- transition baseline: 0.063989
- logistic regression: 0.060968
- HistGradientBoosting: 0.057416

## Main observations
- C1 has substantially narrower mean worst-regret bound G than C0 in this pilot:
  - 10% hidden: C0 0.07867; C1 0.00315.
  - 20% hidden: C0 0.15964; C1 0.01050.
  - 40% hidden: C0 0.31804; C1 0.01850.
- C1 certification rates across the 30 masks:
  - 10% hidden: not certified at epsilon <= 0.0025; certified at epsilon >= 0.01.
  - 20% hidden: not certified at epsilon <= 0.01; certified at epsilon >= 0.02.
  - 40% hidden: not certified at epsilon <= 0.01; certified at epsilon >= 0.02.
- C0 is much less useful under these conditions: at 10% hidden it certifies only at epsilon 0.1; at 20% and 40% hidden it does not certify even at epsilon 0.1.
- No wrong certificates were observed in these 30 seeds. This is a finite pilot, not proof by simulation; C0/C1 exact guarantees come from their mathematical construction under the specified setup.

## Critical limitations
1. The missing labels were artificially hidden from known labels (MCAR); this does not show that naturally missing labels in the source data are MCAR.
2. Masking was performed independently per row. The test set contains repeated observations per `idind` (49,651 rows but 15,917 unique people). If the intended missingness unit is a person rather than a person-year row, rerun with person-level masks as a sensitivity check.
3. The evaluation is on this test split. Do not tune the forecasting models or H1 thresholds using this result.
4. The bridge script retrains the models on `train.csv.gz` using the preprocessing and hyperparameters mirrored from the notebook; it does not load saved model objects.
5. The C2 real-mode output remains scenario sensitivity, not an unconditional guarantee. This artificial-mask grid only runs C0 and C1.
