context("sparse and dense comparisons")

test_that("sparse and dense implementations return equivalent results", {
  set.seed(1)

  # TODO(jolars): test for sparse features with standardization once in-place
  # centering has been implemented

  n <- 100
  p <- 2

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial"),
    intercept = c(TRUE, FALSE),
    alpha = c(0, 0.5, 1),
    standardize = FALSE,
    stringsAsFactors = FALSE
  )

  x_sparse <- Matrix::rsparsematrix(n, p, density = 0.2)
  x_dense <- as.matrix(x_sparse)

  for (i in seq_len(nrow(grid))) {
    pars <- list(
      standardize = FALSE,
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = grid$alpha[i],
      thresh = 1e-5
    )

    pars$y <- switch(pars$family,
                     gaussian = rnorm(n, 10, 2),
                     binomial = rbinom(n, 1, 0.8),
                     multinomial = rbinom(n, 3, 0.5))

    fit_dense <- do.call(sgdnet, modifyList(pars, list(x = x_dense)))
    fit_sparse <- do.call(sgdnet, modifyList(pars, list(x = x_sparse)))

    expect_equivalent(coef(fit_dense), coef(fit_sparse), tol = 1e-2)
  }
})
