#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(haven))
suppressPackageStartupMessages(library(dplyr))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: build_real_j7_benchmark.R INPUT_SAV OUTPUT_DIR')
input <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

vars <- c('idind','year','int_y','adult','age','educ','j1','j7','status','region','psu')
d <- read_sav(input, col_select = tidyselect::all_of(vars), .name_repair = 'minimal')
adult <- as.numeric(d$adult); age <- as.numeric(d$age); educ <- as.numeric(d$educ)
j1 <- as.numeric(d$j1); j7 <- as.numeric(d$j7)
eligible <- adult == 1 & !is.na(j1) & j1 %in% 1:4
unknown <- eligible & !is.na(j7) & j7 >= 1e7
observed <- eligible & !is.na(j7) & j7 < 1e7 & j7 %in% 1:2

make_out <- function(idx, include_y = FALSE) {
  out <- tibble(idind = as.numeric(d$idind[idx]), year = as.numeric(d$year[idx]),
                int_y = as.numeric(d$int_y[idx]), adult = adult[idx], age = age[idx],
                education = educ[idx], current_status = j1[idx],
                settlement = as.numeric(d$status[idx]), region = as.numeric(d$region[idx]),
                psu = as.numeric(d$psu[idx]))
  if (include_y) out$y_j7 <- as.integer(j7[idx] == 1)
  out
}

write.csv(make_out(observed, TRUE), file.path(out_dir, 'real_j7_observed_eligible.csv'), row.names = FALSE, na = '')
u <- make_out(unknown, FALSE)
u$unknown_reason <- 'explicit_refusal_dont_know_no_answer'
u$raw_nonresponse_code <- j7[unknown]
write.csv(u, file.path(out_dir, 'real_j7_unknown_eligible.csv'), row.names = FALSE, na = '')

meta <- tibble(n_total_rows = nrow(d), n_eligible = sum(eligible), n_observed_eligible = sum(observed),
               n_unknown_eligible = sum(unknown), n_structural_no_adult = sum(adult == 0, na.rm = TRUE),
               n_structural_no_current_job = sum(adult == 1 & j1 == 5, na.rm = TRUE),
               outcome = 'j7', outcome_definition = 'worked at main job during previous 30 days',
               unknown_definition = 'adult=1, j1 in 1..4, j7 special refusal/dont-know/no-answer code')
write.csv(meta, file.path(out_dir, 'real_j7_benchmark_metadata.csv'), row.names = FALSE, na = '')
