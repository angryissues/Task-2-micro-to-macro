#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(dplyr))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: evaluate_real_j7_verification.R PROFILE_DIR VERIFICATION_CSV')
profile <- args[[1]]; verification_file <- args[[2]]
obs <- read.csv(file.path(profile, 'real_j7_observed_eligible.csv'))
pred <- read.csv(file.path(profile, 'real_j7_unknown_predictions.csv'))
ver <- read.csv(verification_file, stringsAsFactors=FALSE)
required <- c('idind','year','verified_y_j7')
if (!all(required %in% names(ver))) stop('verification file must contain idind and verified_y_j7')
if (any(!is.na(ver$verified_y_j7) & !ver$verified_y_j7 %in% c(0,1))) stop('verified_y_j7 must be blank, 0, or 1')
dup_key <- paste(ver$idind, ver$year, sep=':')
dup <- dup_key[duplicated(dup_key)]
if (length(dup)) stop('duplicate idind values in verification file')
merged <- pred %>% left_join(ver %>% select(idind, year, verified_y_j7), by=c('idind','year'))
known <- merged %>% filter(!is.na(verified_y_j7))
coverage <- nrow(known) / nrow(pred)
metrics <- data.frame(
  n_unknown_total=nrow(pred), n_verified=nrow(known), verification_coverage=coverage,
  verified_positive_rate=ifelse(nrow(known), mean(known$verified_y_j7), NA),
  brier_baseline=ifelse(nrow(known), mean((known$p_baseline-known$verified_y_j7)^2), NA),
  brier_logistic=ifelse(nrow(known), mean((known$p_logistic-known$verified_y_j7)^2), NA),
  brier_history=ifelse(nrow(known), mean((known$p_history-known$verified_y_j7)^2), NA)
)
out_dir <- dirname(verification_file)
write.csv(metrics, file.path(out_dir, 'real_j7_verification_metrics.csv'), row.names=FALSE)
print(metrics)
