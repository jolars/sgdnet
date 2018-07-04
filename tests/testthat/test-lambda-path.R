context("lambda path computations")

test_that("lambda paths are computed appropriately", {
  library(glmnet)
  glmnet.control(fdev = 0)

  n <- 100
  d <- 3

  # TODO(jolars): test for sparse with standardization once in place centering
  # has been implemented

  for (sparse in c(TRUE, FALSE)) {
    x <- Matrix::rsparsematrix(n, d, density = 0.2)
    if (!sparse)
      x <- as.matrix(x)

    for (alpha in c(0, 0.5, 1)) {
      # TODO(jolars): figure out why glmnet is returning such odd
      # lambda paths when no intercept is fit
      for (intercept in c(TRUE)) {
        for (family in c("gaussian")) {
          y <- switch(family,
                      gaussian = rnorm(n, 5, 2),
                      binomial = rbinom(n, 1, 0.2))

          sfit <- sgdnet(x, y,
                         family = family,
                         alpha = alpha,
                         standardize = !sparse,
                         intercept = intercept)
          gfit <- glmnet(x, y,
                         family = family,
                         alpha = alpha,
                         standardize = !sparse,
                         intercept = intercept)

          expect_equal(sfit$lambda, gfit$lambda)
        }
      }
    }
  }
})
