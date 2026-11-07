#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(dplyr))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: fit_real_j7_models.R PROFILE_DIR OUTPUT_DIR')
profile <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
obs <- read.csv(file.path(profile, 'real_j7_observed_eligible.csv'))
unk <- read.csv(file.path(profile, 'real_j7_unknown_eligible.csv'))
med_educ <- median(obs$education, na.rm = TRUE)
obs$education[is.na(obs$education)] <- med_educ
unk$education[is.na(unk$education)] <- med_educ

rate_by_status <- obs %>% group_by(current_status) %>% summarise(rate=mean(y_j7), .groups='drop')
overall <- mean(obs$y_j7)
lookup <- setNames(rate_by_status$rate, rate_by_status$current_status)
pred_baseline <- function(x) { z <- unname(lookup[as.character(x$current_status)]); z[is.na(z)] <- overall; z }

m_age <- glm(y_j7 ~ age + education + current_status + year, data=obs, family=binomial())
m_simple <- glm(y_j7 ~ age + education + current_status, data=obs, family=binomial())
make_predictions <- function(x) data.frame(
  idind=x$idind, year=x$year, current_status=x$current_status,
  p_baseline=pmin(pmax(pred_baseline(x),0),1),
  p_logistic=pmin(pmax(as.numeric(predict(m_age, newdata=x, type='response')),0),1),
  p_history=pmin(pmax(as.numeric(predict(m_simple, newdata=x, type='response')),0),1))
write.csv(make_predictions(unk), file.path(out_dir, 'real_j7_unknown_predictions.csv'), row.names=FALSE)
obs_pred <- make_predictions(obs)
obs_pred$y_j7 <- obs$y_j7
obs_pred$age <- obs$age
obs_pred$education <- obs$education
write.csv(obs_pred, file.path(out_dir, 'real_j7_observed_predictions.csv'), row.names=FALSE)

set.seed(20261114); take <- sample.int(nrow(obs), floor(.2*nrow(obs)))
fit <- obs[-take,]; hold <- obs[take,]
fit$education[is.na(fit$education)] <- median(fit$education, na.rm=TRUE); hold$education[is.na(hold$education)] <- med_educ
fit_rate <- fit %>% group_by(current_status) %>% summarise(rate=mean(y_j7), .groups='drop')
lk <- setNames(fit_rate$rate, fit_rate$current_status); base <- unname(lk[as.character(hold$current_status)]); base[is.na(base)] <- mean(fit$y_j7)
ma <- glm(y_j7 ~ age + education + current_status + year, data=fit, family=binomial())
ms <- glm(y_j7 ~ age + education + current_status, data=fit, family=binomial())
metrics <- data.frame(model=c('baseline','logistic','history'),
  brier=c(mean((base-hold$y_j7)^2),
          mean((predict(ma,hold,type='response')-hold$y_j7)^2),
          mean((predict(ms,hold,type='response')-hold$y_j7)^2)),
  n_holdout=nrow(hold))
write.csv(metrics, file.path(out_dir, 'real_j7_observed_holdout_metrics.csv'), row.names=FALSE)
