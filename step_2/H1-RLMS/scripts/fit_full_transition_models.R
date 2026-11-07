#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(dplyr))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop('usage: fit_full_transition_models.R TRANSITION_CSV OUTPUT_DIR')
input <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive=TRUE, showWarnings=FALSE)
x <- read.csv(input, stringsAsFactors=FALSE)
x$observed <- !is.na(x$target)
x$target_year <- as.numeric(x$target_year)
last_year <- x %>% group_by(idind) %>% summarise(last_target=max(target_year, na.rm=TRUE), .groups='drop')
person_split <- last_year %>% mutate(split=case_when(last_target<2019~'train', last_target<=2021~'validation', TRUE~'test'))
x <- x %>% left_join(person_split, by='idind')
train <- x %>% filter(split=='train', observed)
emp_fill <- as.numeric(names(sort(table(train$current_employment), decreasing=TRUE)[1]))
train$current_employment[is.na(train$current_employment)] <- emp_fill
x$current_employment[is.na(x$current_employment)] <- emp_fill
for (nm in c('age','sex','education')) {
  med <- median(train[[nm]], na.rm=TRUE)
  train[[nm]][is.na(train[[nm]])] <- med
  x[[nm]][is.na(x[[nm]])] <- med
}
rate <- train %>% group_by(current_employment) %>% summarise(rate=mean(target), .groups='drop')
lookup <- setNames(rate$rate, as.character(rate$current_employment)); overall <- mean(train$target)
x$p_baseline <- unname(lookup[as.character(x$current_employment)]); x$p_baseline[is.na(x$p_baseline)] <- overall
m1 <- glm(target ~ age + sex + education + current_employment, data=train, family=binomial())
m2 <- glm(target ~ age + education + current_employment + target_year, data=train, family=binomial())
x$p_logistic <- pmin(pmax(as.numeric(predict(m1, newdata=x, type='response')),0),1)
x$p_history <- pmin(pmax(as.numeric(predict(m2, newdata=x, type='response')),0),1)
out <- x %>% select(idind,current_year,target_year,age,sex,education,current_employment,target,observed,
                    p_baseline,p_logistic,p_history,current_j1_class,next_j1_class,split)
write.csv(out, file.path(out_dir,'full_transition_model_input.csv'), row.names=FALSE, na='')
metrics <- out %>% filter(observed) %>% group_by(split) %>% summarise(n=n(),
  brier_baseline=mean((p_baseline-target)^2), brier_logistic=mean((p_logistic-target)^2),
  brier_history=mean((p_history-target)^2), .groups='drop')
write.csv(metrics, file.path(out_dir,'full_transition_model_metrics.csv'), row.names=FALSE)
print(metrics)
