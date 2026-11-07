#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(dplyr))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: summarize_real_j7_sensitivity.R PROFILE_DIR OUTPUT_DIR')
profile <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
obs <- read.csv(file.path(profile, 'real_j7_observed_eligible.csv'))
unk <- read.csv(file.path(profile, 'real_j7_unknown_eligible.csv'))
pred <- read.csv(file.path(profile, 'real_j7_unknown_predictions.csv'))
n_obs <- nrow(obs); n_unk <- nrow(unk); n_all <- n_obs + n_unk
obs_pos <- sum(obs$y_j7 == 1)
overall <- mean(obs$y_j7 == 1)
scenarios <- data.frame(
  scenario = c('unknowns_all_no', 'unknowns_all_yes', 'baseline_expected',
               'logistic_expected', 'history_expected'),
  unknown_positive_count = c(0, n_unk, sum(pred$p_baseline),
                             sum(pred$p_logistic), sum(pred$p_history))
)
scenarios$total_positive_rate <- (obs_pos + scenarios$unknown_positive_count) / n_all
scenarios$total_positive_count <- obs_pos + scenarios$unknown_positive_count
scenarios$n_observed <- n_obs
scenarios$n_unknown <- n_unk
scenarios$observed_positive_rate <- overall
write.csv(scenarios, file.path(out_dir, 'real_j7_unknown_sensitivity.csv'), row.names=FALSE)

meta <- data.frame(
  n_observed = n_obs, n_unknown = n_unk, n_total = n_all,
  observed_positive_count = obs_pos, observed_positive_rate = overall,
  worst_case_rate = (obs_pos / n_all),
  best_case_rate = ((obs_pos + n_unk) / n_all),
  unknown_rate = n_unk / n_all
)
write.csv(meta, file.path(out_dir, 'real_j7_unknown_sensitivity_metadata.csv'), row.names=FALSE)
