#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(dplyr))
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop('usage: design_real_j7_verification.R PROFILE_DIR OUTPUT_DIR')
profile <- args[[1]]; out_dir <- args[[2]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
unk <- read.csv(file.path(profile, 'real_j7_unknown_eligible.csv'))

# Conservative normal-approximation design for a finite-population proportion.
# p=.5 maximizes variance; the resulting sample is a planning target, not a
# substitute for observing the missing y_j7 values.
required_n <- function(N, margin, conf=0.95, p=.5) {
  z <- qnorm(1 - (1-conf)/2)
  n0 <- z^2 * p*(1-p) / margin^2
  ceiling(n0 / (1 + (n0 - 1)/N))
}

groups <- bind_rows(
  data.frame(scope='all_unknowns', N=nrow(unk)),
  unk %>% count(current_status, name='N') %>% transmute(scope=paste0('j1=', current_status), N=N)
)
design <- merge(groups, data.frame(margin=c(.05,.10), confidence=.95)) %>%
  mutate(sample_size=mapply(required_n, N, margin, confidence),
         sampling='simple_random_without_replacement',
         variance_assumption='p=0.5')
write.csv(design, file.path(out_dir, 'real_j7_verification_design.csv'), row.names=FALSE)

protocol <- data.frame(
  step=c('draw', 'recontact', 'audit', 'evaluate'),
  requirement=c(
    'Pre-register seed and draw a stratified random sample from real_j7_unknown_eligible.csv; include all j1=2 cases if resources permit.',
    'Obtain the same j7 outcome using an independent recontact or authoritative source; retain outcome provenance and timestamp.',
    'Compare returned outcomes with the original special-missing code; do not overwrite the raw archive.',
    'Use verified labels to evaluate response-model calibration and rerun the C1/nonresponse analysis; treat unverified rows as unknown.'
  )
)
write.csv(protocol, file.path(out_dir, 'real_j7_verification_protocol.csv'), row.names=FALSE)
