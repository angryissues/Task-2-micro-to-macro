#!/usr/bin/env Rscript
suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
})
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop('usage: build_full_h1_transition_from_sav.R INPUT_SAV OUTPUT_CSV')
input_sav <- args[[1]]; output_csv <- args[[2]]

raw <- read_sav(input_sav, col_select=c('idind','year','age','h5','educ','j1'))
num <- function(x) as.numeric(x)
special <- c(99999997, 99999998, 99999999, 99999998.0, 99999999.0)
panel <- raw %>% transmute(idind=num(idind), year=num(year), age=num(age),
  sex=num(h5), education=num(educ), j1_raw=num(j1)) %>%
  mutate(across(c(age, sex, education, j1_raw), ~replace(.x, .x %in% special, NA_real_))) %>%
  arrange(idind, year)

# Match the notebook's binary transition definition: only j1=1 and j1=5.
panel <- panel %>% mutate(employment=case_when(j1_raw==1 ~ 1,
                                                j1_raw==5 ~ 0,
                                                TRUE ~ NA_real_)) %>%
  group_by(idind) %>%
  mutate(next_year=lead(year), next_employment=lead(employment),
         next_age=lead(age), interval=next_year-year,
         age_change=age-lag(age), year_gap=year-lag(year),
         age_discrepancy=age_change-year_gap) %>% ungroup()

age_anomaly_keys <- panel %>% filter(!is.na(year_gap), year_gap>0,
  !is.na(age_discrepancy), abs(age_discrepancy)>2) %>%
  distinct(idind, year) %>% mutate(age_record_suspect=TRUE)

out <- panel %>% filter(interval==1, !is.na(employment), !is.na(next_employment)) %>%
  transmute(idind, current_year=year, target_year=next_year, age, sex, education,
            current_employment=employment, target=as.integer(next_employment)) %>%
  left_join(age_anomaly_keys %>% rename(current_year=year), by=c('idind','current_year')) %>%
  filter(is.na(age_record_suspect)) %>% select(-age_record_suspect)

if (anyDuplicated(out[c('idind','current_year')])) stop('duplicate idind-current_year keys')
if (any(out$target_year-out$current_year != 1)) stop('non-adjacent transition found')
if (any(!out$target %in% c(0,1))) stop('non-binary target found')
dir.create(dirname(output_csv), recursive=TRUE, showWarnings=FALSE)
write.csv(out, output_csv, row.names=FALSE, na='')
cat(sprintf('rows=%d unique_people=%d years=%d-%d\n', nrow(out),
  n_distinct(out$idind), min(out$current_year), max(out$target_year)))
