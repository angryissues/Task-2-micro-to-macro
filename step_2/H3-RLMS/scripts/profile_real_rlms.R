#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(haven))
suppressPackageStartupMessages(library(dplyr))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: profile_real_rlms.R INPUT_SAV OUTPUT_DIR')
input <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

vars <- c('idind', 'year', 'int_y', 'status', 'adult', 'j7', 'j9', 'j11', 'j13', 'j14', 'j15', 'j17', 'j19')
vars <- c(vars, 'j1')
d <- read_sav(input, col_select = tidyselect::all_of(vars), .name_repair = 'minimal')

profile <- bind_rows(lapply(vars[-(1:5)], function(v) {
  x <- as.numeric(d[[v]])
  lab <- attr(d[[v]], 'label'); if (is.null(lab)) lab <- ''
  tibble(variable = v, label = lab,
         n_total = length(x), n_na = sum(is.na(x)),
         n_special_nonresponse = sum(x >= 1e7, na.rm = TRUE),
         n_observed_numeric = sum(!is.na(x) & x < 1e7),
         n_unique_observed = length(unique(x[!is.na(x) & x < 1e7])))
}))
write.csv(profile, file.path(out_dir, 'real_rlms_item_nonresponse_profile.csv'), row.names = FALSE, na = '')

j7 <- as.numeric(d$j7)
adult_vec <- as.numeric(d$adult)
adult_summary <- bind_rows(lapply(sort(unique(adult_vec)), function(a) {
  in_group <- adult_vec == a
  tibble(adult = a, n_total = sum(in_group),
         j7_na = sum(in_group & is.na(j7)),
         j7_special_nonresponse = sum(in_group & j7 >= 1e7, na.rm = TRUE),
         j7_observed = sum(in_group & !is.na(j7) & j7 < 1e7))
}))
write.csv(adult_summary, file.path(out_dir, 'real_rlms_j7_by_adult_questionnaire.csv'), row.names = FALSE)

j1 <- as.numeric(d$j1)
eligible <- adult_vec == 1 & !is.na(j1) & j1 < 1e7 & j1 %in% 1:4
route <- bind_rows(lapply(sort(unique(j1[adult_vec == 1 & !is.na(j1) & j1 < 1e7])), function(v) {
  in_group <- adult_vec == 1 & j1 == v
  tibble(j1 = v, n_total = sum(in_group), j7_na = sum(in_group & is.na(j7)),
         j7_special_nonresponse = sum(in_group & j7 >= 1e7, na.rm = TRUE),
         j7_observed = sum(in_group & !is.na(j7) & j7 < 1e7),
         j7_route = ifelse(v %in% 1:4, 'eligible', 'structural_no_current_job'))
}))
write.csv(route, file.path(out_dir, 'real_rlms_j7_route_by_j1.csv'), row.names = FALSE)

by_year <- d %>% mutate(j7_na = is.na(j7), j7_special = !is.na(j7) & j7 >= 1e7,
                        j7_observed = !is.na(j7) & j7 < 1e7) %>%
  group_by(year) %>% summarise(n_total = n(), n_na = sum(j7_na),
                               n_special_nonresponse = sum(j7_special),
                               n_observed = sum(j7_observed),
                               share_unknown = mean(j7_na | j7_special), .groups = 'drop')
write.csv(by_year, file.path(out_dir, 'real_rlms_j7_nonresponse_by_year.csv'), row.names = FALSE)

meta <- tibble(n_rows = nrow(d), n_unique_idind = n_distinct(d$idind),
               min_year = min(d$year, na.rm = TRUE), max_year = max(d$year, na.rm = TRUE),
               outcome_variable = 'j7', outcome_label = attr(d$j7, 'label'),
               special_code_rule = 'numeric value >= 1e7: refusal/do-not-know/no-answer labels')
write.csv(meta, file.path(out_dir, 'real_rlms_profile_metadata.csv'), row.names = FALSE)
