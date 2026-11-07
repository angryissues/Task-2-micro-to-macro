#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: build_full_real_j7_h3_input.R PROFILE_DIR OUTPUT_DIR')
profile <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
x <- read.csv(file.path(profile, 'real_j7_observed_predictions.csv'))
input <- data.frame(idind=x$idind, target_year=x$year, y=x$y_j7,
                    p_baseline=x$p_baseline, p_logistic=x$p_logistic,
                    p_hist_gradient_boosting=x$p_history)
strata <- data.frame(idind=x$idind, target_year=x$year, age=x$age,
                     education=x$education)
write.csv(input, file.path(out_dir, 'real_j7_full_h3_input.csv'), row.names=FALSE)
write.csv(strata, file.path(out_dir, 'real_j7_full_h3_strata.csv'), row.names=FALSE)
