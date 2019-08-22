test_that("we receive the correct deviance from deviance.sgdnet() with sample weight", {
  set.seed(1)

  library(glmnet)
  glmnet.control(fdev = 0)

  d <- 2
  n <- 100
  x <- matrix(rnorm(n*d), nrow = n, ncol = d)

  loglink <- function(x) {
    pmin <- 1e-9
    pmax <- 1 - pmin
    x <- ifelse(x < pmin, pmin, x)
    x <- ifelse(x > pmax, pmax, x)
    log(x / (1 - x))
  }

  gaussian_nulldev <- function(y, weights) {
    no <- length(y)
    y_center <- (y - weighted.mean(y, weights))^2

    loss <- 0
    for (i in 1:no) {
      loss <- loss + y_center[i]*weights[i]
    }
    loss
  }

  binomial_nulldev <- function(y, intercept = FALSE, weights) {
    no <- length(y)

    if (intercept){
      p <- loglink(weighted.mean(y, weights))
    } else {
      p <- 0 
    }

    loss <- 0
    for (i in 1:no) {
      loss <- loss- (y[i]*p - log(1 + exp(p)))*weights[i]
    }
    2*loss
  }

  multinomial_nulldev <- function(y, intercept = FALSE, weights) {
    no <- length(y)
    nc <- length(unique(y))

    if (intercept) {
      pred <- rep(0, nc)
      for (i in 1:no) {
        pred[y[i]+1] = pred[y[i]+1] + weights[i]/no
      }

    } else {
      pred <- rep(1, nc)/nc
    }

    pred2 <- log(pred) - sum(log(pred))/nc

    loss <- 0
    for (i in seq_len(no)) {
      loss <- loss + (log(sum(exp(pred2))) - pred2[y[i] + 1])*weights[i]
    }

    2*loss
  }

  mgaussian_nulldev <- function(y, weights) {
    no <- nrow(y)
    nc <- ncol(y)

    col_mean <- rep(0, nc)
    for (i in 1:nc) {
      col_mean[i] <- weighted.mean(y[,i], weights)
    }

    y_center <- t(y) - col_mean

    loss <- 0
    for (i in 1:no) {
      loss <- loss + sum(y_center[,i]^2)*weights[i]
    }
    loss
  }
  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial", "mgaussian"),
    intercept = c(TRUE, FALSE),
    alpha = c(0, 0.5, 1),
    standardize = c(TRUE, FALSE),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(grid))) {
    pars <- list(
      x = x,
      standardize = grid$standardize[i],
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = grid$alpha[i],
      thresh = 0.1,
      lambda = 1/nrow(x),
      weights = c(0.5, 1.5, 0.5, 1.5, rep(1, nrow(x)-4))
    )

    y <- switch(pars$family,
                gaussian = rnorm(n, 10, 2),
                binomial = rbinom(n, 1, 0.8),
                multinomial = rbinom(n, 2, 0.5),
                mgaussian = cbind(rnorm(n, 100), rnorm(n)))
    pars$y <- y
    intercept <- pars$intercept

    # compute null deviance manually
    nulldev <- switch(
      pars$family,
      gaussian = gaussian_nulldev(y, weights = weights),
      binomial = binomial_nulldev(y, intercept = intercept, weights = weights),
      multinomial = multinomial_nulldev(y, intercept = intercept, weights = weights),
      mgaussian = mgaussian_nulldev(y, weights = weights)
    )

    sfit <- do.call(sgdnet, pars)
    gfit <- do.call(glmnet, pars)

    expect_equal(sfit$nulldev, gfit$nulldev, tolerance = 1e-6)
    expect_equal(sfit$nulldev, nulldev)
  }
})