#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(dplyr))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: profile_real_j7_panel_history.R PROFILE_DIR OUTPUT_DIR')
profile <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
obs <- read.csv(file.path(profile, 'real_j7_observed_eligible.csv'))
unk <- read.csv(file.path(profile, 'real_j7_unknown_eligible.csv'))

hist <- unk %>% select(idind, year, current_status) %>% rowwise() %>% do({
  u <- .
  h <- obs[obs$idind == u$idind, c('year','y_j7')]
  prev <- h$y_j7[h$year < u$year]
  nextv <- h$y_j7[h$year > u$year]
  data.frame(idind=u$idind, year=u$year, current_status=u$current_status,
             has_previous=length(prev)>0, previous_y=if(length(prev)) prev[which.max(h$year[h$year < u$year])] else NA,
             has_next=length(nextv)>0, next_y=if(length(nextv)) nextv[which.min(h$year[h$year > u$year])] else NA)
}) %>% ungroup()
hist$has_any_neighbor <- hist$has_previous | hist$has_next
hist$neighbor_agrees <- with(hist, has_previous & has_next & previous_y == next_y)
write.csv(hist, file.path(out_dir, 'real_j7_panel_history_by_unknown.csv'), row.names=FALSE, na='')
summary <- data.frame(
  n_unknown=nrow(hist), has_previous=sum(hist$has_previous), has_next=sum(hist$has_next),
  has_any_neighbor=sum(hist$has_any_neighbor), has_both_neighbors=sum(hist$has_previous & hist$has_next),
  neighbors_agree=sum(hist$neighbor_agrees, na.rm=TRUE),
  share_any_neighbor=mean(hist$has_any_neighbor),
  share_both_neighbors=mean(hist$has_previous & hist$has_next)
)
write.csv(summary, file.path(out_dir, 'real_j7_panel_history_summary.csv'), row.names=FALSE)
