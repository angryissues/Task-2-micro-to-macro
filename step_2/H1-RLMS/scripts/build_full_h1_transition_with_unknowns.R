#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(haven); library(dplyr)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop('usage: build_full_h1_transition_with_unknowns.R INPUT_SAV OUTPUT_CSV')
input_sav <- args[[1]]; output_csv <- args[[2]]
raw <- read_sav(input_sav, col_select=c('idind','year','age','h5','educ','adult','j1'))
special <- c(99999997,99999998,99999999)
classify <- function(x) {
 x <- as.numeric(x)
 case_when(is.na(x)~'ordinary_na', x %in% special~'explicit_special_nonresponse',
           x %in% c(1,5)~'binary_state', x %in% c(2,3,4)~'nonbinary_j1_2_4', TRUE~'other_code')
}
p <- raw %>% transmute(idind=as.numeric(idind), year=as.numeric(year), age=as.numeric(age),
 sex=as.numeric(h5), education=as.numeric(educ), adult=as.numeric(adult), j1=as.numeric(j1)) %>%
 mutate(across(c(age,sex,education,j1), ~replace(.x,.x %in% special,NA_real_)),
   j1_class=classify(j1), employment=case_when(j1==1~1,j1==5~0,TRUE~NA_real_)) %>% arrange(idind,year)
nw <- p %>% transmute(idind,next_year=year,next_adult=adult,next_j1_class=j1_class,
 next_employment=employment)
x <- p %>% mutate(next_year=year+1) %>% left_join(nw,by=c('idind','next_year')) %>%
 mutate(age_change=age-lag(age), year_gap=year-lag(year)) %>% group_by(idind) %>%
 mutate(age_change=age-lead(age), year_gap=lead(year)-year) %>% ungroup()
anom <- x %>% filter(!is.na(year_gap),year_gap>0,!is.na(age_change-year_gap),abs(age_change-year_gap)>2) %>%
 distinct(idind,year) %>% mutate(age_record_suspect=TRUE)
out <- x %>% filter(!is.na(next_adult), adult==1, next_adult==1,
  j1_class %in% c('binary_state','explicit_special_nonresponse','ordinary_na'),
  next_j1_class %in% c('binary_state','explicit_special_nonresponse','ordinary_na')) %>%
 transmute(idind,current_year=year,target_year=next_year,age,sex,education,
   current_employment=employment,target=as.integer(next_employment),
   current_j1_class=j1_class,next_j1_class,observed=!is.na(next_employment)) %>%
 left_join(anom %>% rename(current_year=year),by=c('idind','current_year')) %>%
 filter(is.na(age_record_suspect)) %>% select(-age_record_suspect)
if(anyDuplicated(out[c('idind','current_year')])) stop('duplicate keys')
dir.create(dirname(output_csv),recursive=TRUE,showWarnings=FALSE)
write.csv(out,output_csv,row.names=FALSE,na='')
cat(sprintf('rows=%d observed=%d unknown=%d people=%d\n',nrow(out),sum(out$observed),sum(!out$observed),n_distinct(out$idind)))
