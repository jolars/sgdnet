context("deviance")

test_that("we receive the correct deviance from deviance.sgdnet()", {
  set.seed(1)

  library(glmnet)
  glmnet.control(fdev = 0)

  d <- 2
  n <- 1000
  x <- matrix(rnorm(n*d), nrow = n, ncol = d)

  loglink <- function(x) {
    pmin <- 1e-9
    pmax <- 1 - pmin
    x <- ifelse(x < pmin, pmin, x)
    x <- ifelse(x > pmax, pmax, x)
    log(x / (1 - x))
  }

  binomial_nulldev <- function(y, intercept = FALSE) {
    if (intercept)
      p <- loglink(mean(y))
    else
      p <- 0
    -2*sum(y*p - log(1 + exp(p)))
  }

  multinomial_nulldev <- function(y, intercept = FALSE) {
    no <- length(y)
    nc <- length(unique(y))

    if (intercept) {
      pred <- as.vector(prop.table(table(y)))
    } else {
      pred <- rep(1, nc)/nc
    }

    pred2 <- log(pred) - sum(log(pred))/nc

    loss <- 0
    for (i in seq_len(no)) {
      loss <- loss + log(sum(exp(pred2))) - pred2[y[i] + 1]
    }

    2*loss
  }

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial"),
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
      alpha = grid$alpha[i]
    )

    y <- switch(pars$family,
                gaussian = rnorm(n, 10, 2),
                binomial = rbinom(n, 1, 0.8),
                multinomial = rbinom(n, 3, 0.5))
    pars$y <- y
    intercept <- pars$intercept

    # compute null deviance manually
    nulldev <- switch(
      pars$family,
      gaussian = sum((y - mean(y))^2),
      binomial = binomial_nulldev(y, intercept = intercept),
      multinomial = multinomial_nulldev(y, intercept = intercept)
    )

    sfit <- do.call(sgdnet, pars)
    gfit <- do.call(glmnet, pars)

    expect_equal(sfit$nulldev, gfit$nulldev)
    expect_equal(sfit$nulldev, nulldev)
  }
})
