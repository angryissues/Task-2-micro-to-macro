#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(dplyr))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: summarize_real_j7_subgroup_sensitivity.R PROFILE_DIR OUTPUT_DIR')
profile <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
obs <- read.csv(file.path(profile, 'real_j7_observed_eligible.csv'))
unk <- read.csv(file.path(profile, 'real_j7_unknown_eligible.csv'))
o <- obs %>% group_by(current_status) %>% summarise(n_observed=n(), observed_positive=sum(y_j7==1), .groups='drop')
u <- unk %>% count(current_status, name='n_unknown')
out <- full_join(o, u, by='current_status') %>%
  mutate(n_unknown=coalesce(n_unknown,0L), n_total=n_observed+n_unknown,
         observed_rate=observed_positive/n_observed,
         lower_rate=observed_positive/n_total,
         upper_rate=(observed_positive+n_unknown)/n_total,
         unknown_share=n_unknown/n_total) %>% arrange(current_status)
write.csv(out, file.path(out_dir, 'real_j7_subgroup_sensitivity.csv'), row.names=FALSE)
