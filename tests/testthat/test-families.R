context("general family tests")

test_that("all combinations run without errors", {

  library(glmnet)
  glmnet.control(fdev = 0)

  n <- 500
  p <- 2

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial", "mgaussian"),
    intercept = TRUE, # glmnet behaves oddly when the intercept is missing
    alpha = c(0, 0.5, 1),
    standardize = c(TRUE, FALSE),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(grid))) {
    sparse <- grid$sparse[i]

    pars <- list(
      standardize = grid$standardize[i],
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = grid$alpha[i],
      nlambda = 5,
      thresh = 1e-8,
      maxit = 100000
    )

    set.seed(i)

    d <- random_data(n, p, grid$family[i], grid$intercept[i])

    x <- as.matrix(d$x)

    pars$y <- d$y
    pars$x <- x

    sfit <- do.call(sgdnet, pars)
    gfit <- do.call(glmnet, pars)

    compare_predictions(sfit, gfit, x, tol = 1e-3)
  }
})
