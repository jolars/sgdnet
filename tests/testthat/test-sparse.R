context("sparse and dense comparisons")

test_that("sparse and dense implementations return equivalent results", {
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
      thresh = 1e-10,
      nlambda = 5
    )

    set.seed(i)

    d <- random_data(1000, 2, grid$family[i], grid$intercept[i])

    pars$y <- d$y

    set.seed(i)
    fit_sparse <- do.call(sgdnet, modifyList(pars, list(x = d$x)))
    set.seed(i)
    fit_dense <- do.call(sgdnet, modifyList(pars, list(x = as.matrix(d$x))))

    compare_predictions(fit_sparse, fit_dense, d$x, "coefficients", tol = 1e-3)
  }
})
