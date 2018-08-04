context("sparse and dense comparisons")

test_that("sparse and dense implementations return equivalent results", {
  set.seed(1)

  n <- 100
  p <- 2

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial", "mgaussian"),
    intercept = c(TRUE, FALSE),
    alpha = c(0, 0.5, 1),
    standardize = c(TRUE, FALSE),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(grid))) {
    pars <- list(
      standardize = grid$standardize[i],
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = grid$alpha[i],
      nlambda = 20
    )

    x_sparse <- Matrix::rsparsematrix(n, p, density = 0.2)
    x_dense <- as.matrix(x_sparse)

    pars$y <- switch(pars$family,
                     gaussian = rnorm(n, 10, 2),
                     binomial = rbinom(n, 1, 0.8),
                     multinomial = rbinom(n, 3, 0.5),
                     mgaussian = cbind(rnorm(n, -10, 5), rnorm(n, 100, 2)))

    set.seed(i)
    fit_dense <- do.call(sgdnet, modifyList(pars, list(x = x_dense)))
    set.seed(i)
    fit_sparse <- do.call(sgdnet, modifyList(pars, list(x = x_sparse)))

    expect_equivalent(coef(fit_sparse), coef(fit_dense), tol = 1e-3)
  }
})
