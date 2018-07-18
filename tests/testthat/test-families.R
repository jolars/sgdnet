context("general family tests")

test_that("test that all combinations run without errors", {

  library(glmnet)
  glmnet.control(fdev = 0)

  n <- 1000
  p <- 2
  set.seed(1)

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial"),
    intercept = TRUE,
    sparse = FALSE,
    alpha = c(0, 0.5, 1),
    standardize = TRUE,
    stringsAsFactors = FALSE
  )

  x <- Matrix::rsparsematrix(n, p, 0.5)

  for (i in seq_len(nrow(grid))) {
    pars <- list(
      x = if (grid$sparse[i]) x else as.matrix(x),
      standardize = if (grid$sparse[i]) FALSE else grid$standardize[i],
      family = grid$family[i],
      intercept = grid$intercept[i],
      alpha = grid$alpha[i]
    )

    pars$y <- switch(pars$family,
                     gaussian = rnorm(n, 10, 2),
                     binomial = rbinom(n, 1, 0.8),
                     multinomial = rbinom(n, 3, 0.5))

    sfit <- do.call(sgdnet, pars)
    gfit <- do.call(glmnet, pars)

    for (type in c("link", "response", "coefficients")) {
      spred <- predict(sfit, x, type = type)
      gpred <- predict(gfit, x, type = type)
      expect_equivalent(spred, gpred, tolerance = 0.01)
    }

    # treat nonzero predictions and classes separately
    spred <- predict(sfit, x, type = "nonzero")
    gpred <- predict(gfit, x, type = "nonzero")

    f1 <- function(a, b) all(a == b)
    f2 <- function(x, y) mapply(f1, x, y)
    res <- mapply(f2, spred, gpred)
    frac_correct <- sum(unlist(res))/length(res)
    expect_gte(frac_correct, 0.75)

    if (pars$family %in% c("binomial", "multinomial")) {
      spred <- predict(sfit, x, type = "class")
      gpred <- predict(gfit, x, type = "class")
      frac_correct <- sum(spred == gpred)/length(spred)
      expect_gte(frac_correct, 0.99)
    }
  }
})
