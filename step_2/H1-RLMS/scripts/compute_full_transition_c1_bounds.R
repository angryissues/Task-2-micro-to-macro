#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly=TRUE)
if (length(args) < 2) stop('usage: compute_full_transition_c1_bounds.R MODEL_INPUT OUTPUT_DIR')
x <- read.csv(args[[1]]); out_dir <- args[[2]]; dir.create(out_dir,recursive=TRUE,showWarnings=FALSE)
p <- as.matrix(x[,c('p_baseline','p_logistic','p_history')]); y <- x$target; seen <- as.logical(x$observed)
w <- rep(1/nrow(x), nrow(x)); k <- ncol(p)
lo <- matrix(0,k,k); up <- matrix(0,k,k)
for(a in 1:k) for(b in 1:k) if(a!=b) {
  const <- w*(p[,a]^2-p[,b]^2); coeff <- -2*w*(p[,a]-p[,b])
  center <- sum(const) + sum(coeff[seen]*y[seen])
  lo[a,b] <- center + sum(pmin(coeff[!seen],0)); up[a,b] <- center + sum(pmax(coeff[!seen],0))
}
g <- apply(up,1,max); rec <- which.min(g)-1
dec <- data.frame(method='C1',n_rows=nrow(x),n_observed=sum(seen),n_unknown=sum(!seen),
  recommended_model=rec,G=min(g),certified_0025=min(g)<=.0025,
  certified_005=min(g)<=.005,certified_01=min(g)<=.01)
write.csv(dec,file.path(out_dir,'full_transition_c1_decision.csv'),row.names=FALSE)
pair <- expand.grid(model_a=0:(k-1),model_b=0:(k-1)); pair$lower=mapply(function(a,b)lo[a+1,b+1],pair$model_a,pair$model_b); pair$upper=mapply(function(a,b)up[a+1,b+1],pair$model_a,pair$model_b)
write.csv(pair,file.path(out_dir,'full_transition_c1_pairwise_bounds.csv'),row.names=FALSE)
print(dec)
