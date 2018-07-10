
# Preprepared benchmark data for vignette ---------------------------------

binomial_loss <- function(fit, X, y, lambda, alpha) {
  # tidy up data
  n <- NROW(X)
  X <- t(X)
  y <- as.vector(as.numeric(y))
  y[y == min(y)] <- 0
  y[y > min(y)] <- 1
  beta <- fit$beta
  beta0 <- as.vector(fit$a0)

  n <- length(y)
  # binomial loglikelihood
  cXb <- crossprod(X, beta)
  loglik <- sum(y*(beta0 + cXb) - log(1 + exp(beta0 + cXb)))

  # compute penalty
  penalty <- 0.5*(1 - alpha)*sum(beta^2) + alpha*sum(abs(beta))
  -loglik/n + lambda*penalty
}

library(SparseM)
library(Matrix)
library(glmnet)
library(gsoc18saga) # https://github.com/jolars/gsoc18saga/

# load datasets
datasets <- list(
  mushrooms = mushrooms,
  covtype = covtype,
  a9a = a9a,
  phishing = phishing,
  ijcnn1 = ijcnn1
)

# setup tolerance sequence to iterate over
n_tol <- 100
sgdnet_tol <- signif(exp(seq(log(0.95), log(1e-3), length.out = n_tol)), 2)
glmnet_tol <- signif(exp(seq(log(0.95), log(1e-6), length.out = n_tol)), 2)

# setup result data.frame
data_binomial <- data.frame(dataset = character(),
                            package = character(),
                            penalty = character(),
                            time = double(),
                            loss = double())

# compute timings
for (i in seq_along(datasets)) {
  cat(names(datasets)[i], "\n")
  X <- as(datasets[[i]]$x, "dgCMatrix")
  y <- datasets[[i]]$y
  n_obs <- nrow(X)

  lambda <- 1/n_obs

  for (j in seq_len(n_tol)) {
    set.seed(j*i)
    cat("\r", j, "/", n_tol)
    flush.console()
    if (j == n_tol)
      cat("\n")

    for (penalty in c("ridge", "lasso")) {
      alpha <- ifelse(penalty == "ridge", 0, 1)
      glmnet_time <- system.time({
        glmnet_fit <- glmnet::glmnet(X,
                                     y,
                                     family = "binomial",
                                     lambda = lambda,
                                     alpha = alpha,
                                     intercept = TRUE,
                                     standardize = FALSE,
                                     thresh = glmnet_tol[j])
      })

      sgdnet_time <- system.time({
        sgdnet_fit <- sgdnet::sgdnet(X,
                                     y,
                                     family = "binomial",
                                     lambda = lambda,
                                     alpha = alpha,
                                     intercept = TRUE,
                                     standardize = FALSE,
                                     thresh = sgdnet_tol[j])
      })

      # retrieve loss
      glmnet_loss <- binomial_loss(glmnet_fit, X, y, lambda = lambda, alpha = alpha)
      sgdnet_loss <- binomial_loss(sgdnet_fit, X, y, lambda = lambda, alpha = alpha)

      data_binomial <- rbind(data_binomial,
                             data.frame(
                               dataset = names(datasets)[i],
                               package = c("glmnet", "sgdnet"),
                               time = c(glmnet_time[3], sgdnet_time[3]),
                               loss = c(glmnet_loss, sgdnet_loss),
                               penalty = penalty
                             ))
    }
  }
}

library(tidyverse)
benchmarks_binomial <- data_binomial[order(data_binomial$time), ]

usethis::use_data(benchmarks_binomial, overwrite = TRUE)

