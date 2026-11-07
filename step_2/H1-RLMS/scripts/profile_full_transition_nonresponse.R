#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(haven); library(dplyr)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop('usage: profile_full_transition_nonresponse.R INPUT_SAV OUTPUT_DIR')
input_sav <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
raw <- read_sav(input_sav, col_select=c('idind','year','adult','j1'))
special <- c(99999997,99999998,99999999)
classify <- function(x) {
  x <- as.numeric(x)
  case_when(is.na(x) ~ 'ordinary_na', x %in% special ~ 'explicit_special_nonresponse',
            x %in% c(1,5) ~ 'binary_state', x %in% c(2,3,4) ~ 'nonbinary_j1_2_4',
            TRUE ~ 'other_code')
}
p <- raw %>% transmute(idind=as.numeric(idind), year=as.numeric(year),
  adult=as.numeric(adult), j1=as.numeric(j1)) %>%
  mutate(j1_class=classify(j1), binary=case_when(j1==1~1, j1==5~0, TRUE~NA_real_)) %>%
  arrange(idind,year)
next_wave <- p %>% transmute(idind, next_year=year, next_adult=adult,
  next_j1=j1, next_class= j1_class, next_binary=binary)
x <- p %>% mutate(next_year=year+1) %>% left_join(next_wave, by=c('idind','next_year')) %>%
  mutate(has_next_wave=!is.na(next_adult),
    pair_class=case_when(!has_next_wave~'no_consecutive_next_wave',
      adult==0 | next_adult==0~'structural_no_adult_questionnaire',
      j1_class=='explicit_special_nonresponse' | next_class=='explicit_special_nonresponse'~'explicit_item_nonresponse',
      j1_class=='ordinary_na' | next_class=='ordinary_na'~'ordinary_item_missing',
      j1_class=='nonbinary_j1_2_4' | next_class=='nonbinary_j1_2_4'~'nonbinary_route_j1_2_4',
      j1_class=='binary_state' & next_class=='binary_state'~'observed_binary_transition',
      TRUE~'other_or_unclassified'))

summary <- x %>% count(pair_class, name='n') %>% mutate(share=n/sum(n)) %>% arrange(desc(n))
write.csv(summary, file.path(out_dir,'full_transition_nonresponse_summary.csv'), row.names=FALSE)
by_year <- x %>% count(year, pair_class, name='n') %>% group_by(year) %>% mutate(share=n/sum(n)) %>% ungroup()
write.csv(by_year, file.path(out_dir,'full_transition_nonresponse_by_year.csv'), row.names=FALSE)
meta <- data.frame(n_person_wave=nrow(p), n_people=n_distinct(p$idind),
  n_observed_binary=sum(x$pair_class=='observed_binary_transition'),
  n_explicit_nonresponse=sum(x$pair_class=='explicit_item_nonresponse'),
  n_ordinary_missing=sum(x$pair_class=='ordinary_item_missing'),
  n_nonbinary_route=sum(x$pair_class=='nonbinary_route_j1_2_4'),
  n_no_next_wave=sum(x$pair_class=='no_consecutive_next_wave'))
write.csv(meta, file.path(out_dir,'full_transition_nonresponse_metadata.csv'), row.names=FALSE)
print(meta)
