#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(dplyr))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: rank_real_j7_verification_queue.R PROFILE_DIR OUTPUT_DIR')
profile <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
h <- read.csv(file.path(profile, 'real_j7_panel_history_by_unknown.csv'))
p <- read.csv(file.path(profile, 'real_j7_unknown_predictions.csv'))
out <- h %>% left_join(p, by=c('idind','year','current_status')) %>%
  mutate(model_uncertainty = abs(p_logistic - 0.5),
         priority_group = case_when(
           !has_any_neighbor ~ 'P1_no_panel_history',
           has_previous & has_next & !neighbor_agrees ~ 'P1_conflicting_neighbors',
           xor(has_previous, has_next) ~ 'P2_one_neighbor',
           has_previous & has_next & neighbor_agrees ~ 'P3_agreeing_neighbors',
           TRUE ~ 'P2_unclassified'),
         priority = case_when(grepl('P1', priority_group) ~ 1L,
                              grepl('P2', priority_group) ~ 2L, TRUE ~ 3L)) %>%
  arrange(priority, desc(model_uncertainty), year, idind)
out$queue_rank <- seq_len(nrow(out))
write.csv(out, file.path(out_dir, 'real_j7_verification_queue.csv'), row.names=FALSE, na='')
summary <- out %>% count(priority, priority_group, name='n') %>% arrange(priority, priority_group)
write.csv(summary, file.path(out_dir, 'real_j7_verification_queue_summary.csv'), row.names=FALSE)
