#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: prepare_real_j7_verification_template.R PROFILE_DIR OUTPUT_DIR')
profile <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
u <- read.csv(file.path(profile, 'real_j7_unknown_eligible.csv'), stringsAsFactors=FALSE)
template <- u[, c('idind','year','current_status','unknown_reason','raw_nonresponse_code')]
template$verified_y_j7 <- NA_integer_
template$verification_source <- NA_character_
template$verification_timestamp <- NA_character_
template$verifier_note <- NA_character_
write.csv(template, file.path(out_dir, 'real_j7_verification_template.csv'), row.names=FALSE, na='')
