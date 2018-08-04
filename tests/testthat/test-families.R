context("general family tests")

test_that("test that all combinations run without errors", {

  library(glmnet)
  glmnet.control(fdev = 0)

  n <- 1000
  p <- 2

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial", "mgaussian"),
    intercept = TRUE, # glmnet behaves oddly when the intercept is missing
    sparse = c(TRUE, FALSE),
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
      nlambda = 30,
      thresh = 1e-4
    )

    set.seed(i)

    x <- Matrix::rsparsematrix(n, p, 0.2)

    if (!sparse)
      x <- as.matrix(x)

    pars$x <- x

    pars$y <- switch(pars$family,
                     gaussian = rnorm(n, 10, 2),
                     binomial = rbinom(n, 1, 0.8),
                     multinomial = rbinom(n, 3, 0.5),
                     mgaussian = matrix(rnorm(n*2), n, 2))

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

    gpred <- lapply(gpred, function(x) if (is.null(x)) NA else x)
    spred <- lapply(spred, function(x) if (is.null(x)) NA else x)

    f1 <- function(a, b) all(a == b)
    f2 <- function(x, y) mapply(f1, x, y)
    res <- mapply(f2, spred, gpred)
    frac_correct <- sum(unlist(res[!is.na(res)]))/length(unlist(res))
    expect_gte(frac_correct, 0.75)

    if (pars$family %in% c("binomial", "multinomial")) {
      spred <- predict(sfit, x, type = "class")
      gpred <- predict(gfit, x, type = "class")
      frac_correct <- sum(unlist(spred) == unlist(gpred))/length(unlist(spred))
      expect_gte(frac_correct, 0.96)
    }
  }
})
