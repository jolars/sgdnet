
# Preprepared benchmark data for vignette ---------------------------------

# Normalize, cut, and aggregate benchmark data
aggregate_benchmarks <- function(x) {
  library(tidyverse)
  x <- x %>%
    group_by(package, penalty, dataset) %>%
    mutate(min_pkg_time = min(time),
           max_pkg_time = max(time)) %>%
    ungroup() %>%
    group_by(dataset, penalty) %>%
    mutate(min_set_time = max(min_pkg_time),
           max_set_time = min(max_pkg_time)) %>%
    ungroup() %>%
    filter(time >= min_set_time, time <= max_set_time) %>%
    mutate(time_rel = (time - min_set_time)/(max_set_time - min_set_time)) %>%
    mutate(time_cut = cut(time_rel, seq(0, 1.0, by = 0.05))) %>%
    group_by(dataset, penalty) %>%
    mutate(min_loss = min(loss),
           max_loss = max(loss)) %>%
    ungroup() %>%
    mutate(normalized_loss = (loss - min_loss)/(max_loss - min_loss)) %>%
    group_by(dataset, penalty, package, time_cut) %>%
    summarise(normalized_mean_loss = mean(normalized_loss)) %>%
    mutate(time = as.numeric(time_cut)/20) %>%
    select(dataset = dataset,
           penalty = penalty,
           package = package,
           time = time,
           loss = normalized_mean_loss)
  x
}
#
# # loss for the binomial case
# binomial_loss <- function(fit, X, y, lambda, alpha) {
#   # tidy up data
#   n <- NROW(X)
#   X <- t(X)
#   y <- as.vector(as.numeric(y))
#   y[y == min(y)] <- 0
#   y[y > min(y)] <- 1
#   beta <- fit$beta
#   beta0 <- as.vector(fit$a0)
#
#   n <- length(y)
#   # binomial loglikelihood
#   cXb <- crossprod(X, beta)
#   loglik <- sum(y*(beta0 + cXb) - log(1 + exp(beta0 + cXb)))
#
#   # compute penalty
#   penalty <- 0.5*(1 - alpha)*sum(beta^2) + alpha*sum(abs(beta))
#   -loglik/n + lambda*penalty
# }

benchmark <- function(datasets, family, n = 100) {
  library(sgdnet)
  library(glmnet)
  library(SparseM)
  library(Matrix)

  # setup tolerance sequence to iterate over
  sgdnet_tol <- signif(exp(seq(log(0.1), log(1e-10), length.out = n)), 2)
  glmnet_tol <- signif(exp(seq(log(0.1), log(1e-6), length.out = n)), 2)

  # iter <- seq_len(n)

  for (i in seq_along(datasets)) {
    cat(names(datasets)[i], "\n")
    X <- datasets[[i]]$x
    y <- datasets[[i]]$y
    n_obs <- nrow(X)

    lambda <- 1/n_obs

    data <-  data.frame(dataset = character(),
                        package = character(),
                        penalty = character(),
                        time = double(),
                        loss = double())

    for (j in seq_len(n)) {
      set.seed(j*i)
      cat("\r", j, "/", n)
      flush.console()
      if (j == n)
        cat("\n")

      for (penalty in c("ridge", "lasso")) {
        alpha <- ifelse(penalty == "ridge", 0, 1)
        glmnet_time <- system.time({
          glmnet_fit <- glmnet::glmnet(X,
                                       y,
                                       family = family,
                                       lambda = lambda,
                                       alpha = alpha,
                                       intercept = TRUE,
                                       standardize = FALSE,
                                       maxit = n)
        })

        sgdnet_time <- system.time({
          sgdnet_fit <- sgdnet::sgdnet(X,
                                       y,
                                       family = family,
                                       lambda = lambda,
                                       alpha = alpha,
                                       intercept = TRUE,
                                       standardize = FALSE,
                                       maxit = n)
        })

        glmnet_loss <- glmnet_fit$dev.ratio
        sgdnet_loss <- sgdnet_fit$dev.ratio

        data <- rbind(data,
                      data.frame(
                        dataset = names(datasets)[i],
                        package = c("glmnet", "sgdnet"),
                        time = c(glmnet_time[3], sgdnet_time[3]),
                        time = c(0, 0),
                        loss = c(0, 0),
                        loss = c(glmnet_loss, sgdnet_loss),
                        penalty = penalty
                      ))
      }
    }
  }
  data
}

# Gaussian ----------------------------------------------------------------

library(libsvmdata) # https://github.com/jolars/libsvmdata/

bodyfat <- getData("bodyfat", scaled = TRUE)
cadata <- getData("cadata")
cadata$x <- scale(cadata$x)
cpusmall <- getData("cpusmall", scaled = TRUE)

gaussian_datasets <- list(bodyfat = bodyfat,
                          cadata = cadata,
                          cpusmall = cpusmall)

d <- benchmark(gaussian_datasets, "gaussian")
benchmarkdata_gaussian <- aggregate_benchmarks(d)

usethis::use_data(benchmarkdata_gaussian, overwrite = TRUE, internal = TRUE)

# Binomial ----------------------------------------------------------------

mushrooms <- getData("mushrooms")
a9a <- getData("a9a", "training")
ijcnn1 <- getData("ijcnn1", "training", scaled = TRUE)
ijcnn1$x <- scale(ijcnn1$x)

binomial_datasets <- list(mushrooms = mushrooms,
                          adult = a9a,
                          ijcnn1 = ijcnn1)

d <- benchmark(binomial_datasets, "binomial")
benchmarkdata_binomial <- aggregate_benchmarks(d)

usethis::use_data(benchmarkdata_binomial, overwrite = TRUE, internal = TRUE)

# Multinomial -------------------------------------------------------------

poker <- getData("poker", "training", scaled = FALSE)
poker$x <- scale(poker$x)
satimage <- getData("satimage", scaled = TRUE)
pendigits <- getData("pendigits", "training")

multinomial_datasets <- list(poker = poker,
                             satimage = satimage,
                             pendigits = pendigits)

d <- benchmark(multinomial_datasets, "multinomial")
benchmarkdata_multinomial <- aggregate_benchmarks(d)

benchmarks <- list(gaussian = benchmarkdata_gaussian,
                   binomial = benchmarkdata_binomial,
                   multinomial = benchmarkdata_multinomial)

usethis::use_data(benchmarks, overwrite = TRUE, internal = TRUE)
