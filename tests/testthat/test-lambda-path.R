context("lambda path")

test_that("lambda paths are computed as in glmnet", {
  library(glmnet)
  glmnet.control(fdev = 0)

  # TODO(jolars): test for sparse features with standardization once in-place
  # centering has been implemented

  n <- 100
  p <- 2
  set.seed(1)

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial", "mgaussian"),
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
      standardize = grid$standardize[i],
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = grid$alpha[i],
      thresh = 1e-1, # we only care about the regularization paths
      nlambda = 10
    )

    pars$y <- switch(pars$family,
                     gaussian = rnorm(n, 10, 2),
                     binomial = rbinom(n, 1, 0.8),
                     multinomial = rbinom(n, 2, 0.5),
                     mgaussian = cbind(rnorm(n), rnorm(n, -19)))

    sfit <- do.call(sgdnet, pars)
    gfit <- do.call(glmnet, pars)

    expect_equal(sfit$lambda, gfit$lambda)
  }
})

test_that("lambda path checks out against manual calculations", {
  set.seed(1)

  lambda_max <- function(x,
                         y,
                         family = c("gaussian",
                                    "binomial",
                                    "multinomial",
                                    "mgaussian"),
                         standardize = TRUE,
                         alpha = 1,
                         ...) {
    family <- match.arg(family)

    sd2 <- function(x) sqrt(sum((x - mean(x))^2)/length(x))

    if (standardize) {
      x2 <- as.matrix(scale(x, scale = apply(x, 2, sd2)))
    } else {
      x2 <- as.matrix(x)
    }

    if (family == "binomial") {
      y2 <- as.numeric(as.factor(y)) - 1
    } else if (family == "multinomial") {
      m <- length(unique(y))
      y2 <- matrix(0, nrow = nrow(x), ncol = m)
      yy <- as.numeric(as.factor(y))

      for (i in seq_len(nrow(x))) {
        y2[i, yy[i]] <- 1
      }
    } else {
      y2 <- y
    }

    ys <- apply(as.matrix(y2), 2, sd2)
    y3 <- scale(y2, scale = ys)

    inner_products <- crossprod(x2, y3)
    for (i in seq_len(ncol(inner_products)))
      inner_products[, i] = inner_products[, i]*ys[i]

    if (family == "multinomial") {
      max(abs(inner_products))/(nrow(x)*max(alpha, 1e-3))
    } else {
      max(sqrt(rowSums(inner_products^2)))/(nrow(x)*max(alpha, 1e-3))
    }
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
      x = as.matrix(subset(mtcars, select = c("cyl", "disp", "hp", "am"))),
      standardize = grid$standardize[i],
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = grid$alpha[i]
    )

    pars$y <- switch(grid$family[i],
                     gaussian = mtcars$mpg,
                     binomial = mtcars$vs,
                     multinomial = mtcars$gear,
                     mgaussian = cbind(mtcars$hp, mtcars$drat))

    fit <- do.call(sgdnet, pars)

    sgdnet_lambda <- max(fit$lambda)
    manual_lambda <- do.call(lambda_max, pars)

    expect_equal(sgdnet_lambda, manual_lambda)
  }
})

test_that("the first lasso fit is sparse", {
  set.seed(1)

  expect_sparse_start <- function(x) {
    all(abs(x[, 1]) - 1e-5 < 0)
  }

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial", "mgaussian"),
    intercept = c(TRUE, FALSE),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(grid))) {
    pars <- list(
      x = as.matrix(subset(mtcars, select = c("cyl", "disp", "hp", "am"))),
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = 1
    )

    pars$y <- switch(grid$family[i],
                     gaussian = mtcars$mpg,
                     binomial = mtcars$vs,
                     multinomial = mtcars$gear,
                     mgaussian = cbind(mtcars$hp, mtcars$drat))

    fit <- do.call(sgdnet, pars)

    # Check that first solution is completely sparse
    if (pars$alpha  == 1) {
      if (pars$family %in% c("multinomial", "mgaussian")) {
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
    family = c("gaussian", "binomial", "multinomial", "mgaussian"),
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
                     multinomial = mtcars$gear,
                     mgaussian = cbind(mtcars$hp, mtcars$drat))
    set.seed(1)
    fit1 <- do.call(sgdnet, pars)
    set.seed(1)
    fit2 <- do.call(sgdnet, modifyList(pars, list(lambda = fit1$lambda)))

    expect_equal(coef(fit1), coef(fit2))
  }
})
