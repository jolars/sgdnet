context("sparse and dense comparisons")
source("helpers.R")

test_that("sparse and dense implementations return equivalent results", {
  set.seed(1)

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
      nlambda = 5,
      thresh = 1e-4
    )

    d <- random_data(300, 3, grid$family[i], grid$intercept[i])

    pars$y <- d$y

    set.seed(i)
    fit_sparse <- do.call(sgdnet, modifyList(pars, list(x = d$x)))
    set.seed(i)
    fit_dense <- do.call(sgdnet, modifyList(pars, list(x = as.matrix(d$x))))

    compare_predictions(fit_sparse, fit_dense, d$x, "coefficients", tol = 1e-1)
  }
})
