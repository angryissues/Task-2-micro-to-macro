#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(haven); library(dplyr)})
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop('usage: profile_rlms_trajectories.R INPUT_SAV OUTPUT_DIR')
input_sav <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
special <- c(99999997,99999998,99999999)
p <- read_sav(input_sav, col_select=c('idind','year','adult','j1','age','h5','educ')) %>%
  transmute(idind=as.numeric(idind), year=as.numeric(year), adult=as.numeric(adult), j1=as.numeric(j1),
            age=as.numeric(age), sex=as.numeric(h5), education=as.numeric(educ)) %>%
  mutate(j1=replace(j1,j1 %in% special,NA_real_),
         employment=case_when(j1==1~1,j1==5~0,TRUE~NA_real_)) %>% arrange(idind,year)

# Three annual future control points: t+1, t+2, t+3. Only complete binary
# trajectories enter the first observed-path benchmark.
base <- p %>% select(idind, year, adult, employment, age, sex, education)
out <- base %>% rename(start_year=year, start_adult=adult, start_employment=employment) %>%
  left_join(base %>% transmute(idind, y1=year-1, adult1=adult, e1=employment), by=c('idind','start_year'='y1')) %>%
  left_join(base %>% transmute(idind, y2=year-2, adult2=adult, e2=employment), by=c('idind','start_year'='y2')) %>%
  left_join(base %>% transmute(idind, y3=year-3, adult3=adult, e3=employment), by=c('idind','start_year'='y3')) %>%
  mutate(complete_path=adult1==1 & adult2==1 & adult3==1 & !is.na(e1) & !is.na(e2) & !is.na(e3),
         first_exit_h1=as.integer(e1==0), first_exit_h2=as.integer(e1==0 | e2==0),
         first_exit_h3=as.integer(e1==0 | e2==0 | e3==0))

meta <- data.frame(n_person_wave=nrow(p), n_people=n_distinct(p$idind),
  n_windows=nrow(out), n_complete_paths=sum(out$complete_path,na.rm=TRUE),
  n_incomplete_paths=sum(!out$complete_path,na.rm=TRUE), h=3)
write.csv(meta,file.path(out_dir,'rlms_trajectory_metadata.csv'),row.names=FALSE)
summary <- out %>% mutate(path_class=case_when(complete_path~'complete_binary_path',
  is.na(adult1)|is.na(adult2)|is.na(adult3)~'missing_wave_or_adult',
  TRUE~'nonbinary_or_item_missing')) %>% count(path_class,name='n') %>% mutate(share=n/sum(n))
write.csv(summary,file.path(out_dir,'rlms_trajectory_summary.csv'),row.names=FALSE)
complete <- out %>% filter(complete_path) %>% count(start_year, name='n_complete')
write.csv(complete,file.path(out_dir,'rlms_complete_paths_by_start_year.csv'),row.names=FALSE)
write.csv(out %>% filter(complete_path),file.path(out_dir,'rlms_complete_trajectory_windows.csv'),row.names=FALSE,na='')
print(meta); print(summary)
