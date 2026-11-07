# H3 real-data plan: RLMS individual panel

## Identified source and outcome

The archive `RLMS_IND_1994_2025_rus.sav` contains 474,600 individual-wave
records, 62,841 distinct `idind` values and 30 survey years. The primary
employment item is `j7`: whether the respondent worked at the main job during
the last 30 days.

The source stores both ordinary missing values (`NA`) and special nonresponse
codes (numeric values around `1e8`, labelled refusal/no-answer/don't-know).
These must not be collapsed with structural skip patterns.

The first profile already shows why this matters: 81,218 records have no adult
questionnaire (`adult=0`), and all of their `j7` values are structurally empty.
Among records with an adult questionnaire, only 224 `j7` values use explicit
special nonresponse codes; the remaining missing values require the questionnaire
routing rule before they can be called nonresponse.

The routing check resolves this for the primary item: `j1=1..4` means the
respondent currently works or is on a work-related leave, and `j7` is eligible.
For these 212,594 eligible records, `j7` has 212,370 observed answers and 224
explicit nonresponse codes, with zero ordinary `NA`. `j1=5` means no current job;
its 180,467 empty `j7` values are structural skips, not nonresponse.

## First real-incomplete layer

The profile script keeps `NA`, special nonresponse and observed numeric answers
separate. Its first output is an item-nonresponse benchmark for `j7`, not yet a
full attrition benchmark. A person absent from a wave is a different missingness
event and requires a wave/household eligibility rule.

## Required next decision

Before fitting H3 on real unknowns, freeze the coding rule for `j7`:

- observed outcome: valid substantive values only;
- item nonresponse: explicit refusal/no-answer/don't-know codes;
- structural skip: kept separate and excluded from the response denominator;
- absent person-wave: not inferred as a nonresponse until eligibility is known.

For the first real H3 benchmark, define `j7_unknown` only as the explicit
refusal/no-answer/don't-know codes among `adult=1, j1 in {1,2,3,4}`. Keep
`j1=5` and `adult=0` outside the response denominator.

## First real-incomplete benchmark

The extraction now writes:

- `real_j7_observed_eligible.csv`: 212,370 observed labels for model fitting;
- `real_j7_unknown_eligible.csv`: 224 explicit unknown cases, with no imputed
  outcome;
- `real_j7_unknown_predictions.csv`: frozen exploratory predictions from three
  candidate models;
- `real_j7_observed_holdout_metrics.csv`: a 20% observed-label holdout check.

The holdout Brier scores were 0.02565 for the status-rate baseline, 0.02877 for
the logistic model with age/education/status/year, and 0.02876 for the simpler
history model. This is a model-screening diagnostic, not a claim that the
unknown cases have been recovered: their `y_j7` remains missing.
