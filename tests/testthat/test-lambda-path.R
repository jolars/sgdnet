context("lambda path")

test_that("lambda paths are computed as in glmnet", {
  library(glmnet)
  glmnet.control(fdev = 0)

  # TODO(jolars): test for sparse features with standardization once in-place
  # centering has been implemented

  n <- 1000
  p <- 2
  set.seed(1)

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial"),
    intercept = TRUE,
    sparse = c(TRUE, FALSE),
    alpha = c(0, 0.5, 1),
    standardize = c(TRUE, FALSE),
    stringsAsFactors = FALSE
  )

  x <- Matrix::rsparsematrix(n, p, 0.5)

  for (i in seq_len(nrow(grid))) {
    pars <- list(
      x = if (grid$sparse[i]) x else as.matrix(x),
      standardize = if (grid$sparse[i]) FALSE else grid$standardize[i],
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = grid$alpha[i],
      thresh = 1e-1 # we only care about the regularization paths
    )

    pars$y <- switch(pars$family,
                     gaussian = rnorm(n, 10, 2),
                     binomial = rbinom(n, 1, 0.8),
                     multinomial = rbinom(n, 3, 0.5))

    sfit <- do.call(sgdnet, pars)
    gfit <- do.call(glmnet, pars)

    expect_equal(sfit$lambda, gfit$lambda)
  }
})

test_that("lambda path checks out against manual calculations", {
  set.seed(1)

  lambda_max <- function(x, y,
                         family = c("gaussian", "binomial", "multinomial"),
                         standardize = TRUE, alpha = 1, ...) {
    family <- match.arg(family)

    sd2 <- function(x) sqrt(sum((x - mean(x))^2)/length(x))
    if (standardize) {
      x2 <- as.matrix(scale(x, scale = apply(x, 2, sd2)))
    } else {
      x2 <- as.matrix(x)
    }

    m <- length(unique(y))

    if (family %in% c("binomial", "multinomial")) {
      y2 <- matrix(0, nrow = nrow(x), ncol = m)
      yy <- as.numeric(as.factor(y))

      for (i in seq_len(nrow(x))) {
        y2[i, yy[i]] <- 1
      }

      ys <- apply(y2, 2, sd2)
      y3 <- scale(y2, scale = ys)
    } else {
      ys <- sd2(y)
      y3 <- as.matrix((y - mean(y))/ys)
    }

    max(abs(crossprod(y3, x2)*ys))/(nrow(x)*max(alpha, 1e-3))
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
      x = subset(mtcars, select = c("cyl", "disp", "hp", "am")),
      standardize = grid$standardize[i],
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = grid$alpha[i]
    )

    pars$y <- switch(grid$family[i],
                     gaussian = mtcars$mpg,
                     binomial = mtcars$vs,
                     multinomial = mtcars$gear)

    fit <- do.call(sgdnet, pars)

    sgdnet_lambda <- max(fit$lambda)
    manual_lambda <- do.call(lambda_max, pars)

    expect_equal(sgdnet_lambda, manual_lambda)

    # Check that first solution is completely sparse
    expect_sparse_start <- function(x) {
      all(abs(x[, 1]) - 1e-3 < 0)
    }

    if (pars$alpha  == 1) {
      if (pars$family %in% c("multinomial")) {
        sparse_starts <- sapply(fit$beta, expect_sparse_start)
        expect_true(any(sparse_starts))
      } else {
        expect_true(expect_sparse_start(fit$beta))
      }
    }
  }
})


test_that("refitting model with automatically generated path gives same fit", {
  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial"),
    standardize = c(TRUE, FALSE),
    stringsAsFactors = FALSE
  )

  pars <- list(y = NULL,
               x = subset(mtcars, select = c("cyl", "disp", "hp", "am")))

  for (i in seq_len(nrow(grid))) {
    pars <- modifyList(pars, list(family = grid$family[i],
                                  standardize = grid$standardize[i]))
    pars$y <- switch(grid$family[i],
                     gaussian = mtcars$mpg,
                     binomial = mtcars$vs,
                     multinomial = mtcars$gear)
    set.seed(1)
    fit1 <- do.call(sgdnet, pars)
    set.seed(1)
    fit2 <- do.call(sgdnet, modifyList(pars, list(lambda = fit1$lambda)))

    expect_equal(coef(fit1), coef(fit2))
  }

})
