context("binomial regression")

test_that("basic model fitting with dense and sparse features", {
  library(glmnet)
  glmnet.control(fdev = 0)

  set.seed(2)

  n <- 1000
  d <- 3

  # Sparse features
  x <- Matrix::rsparsematrix(n, d, density = 0.2)
  y <- rbinom(n, 1, 0.5)

  for (alpha in c(0, 0.5, 1)) {
    sfit_sparse <- sgdnet(x, y, alpha = alpha,
                          family = "binomial",
                          standardize = FALSE,
                          intercept = FALSE)
    sfit_dense <- sgdnet(as.matrix(x), y, alpha = alpha,
                         family = "binomial",
                         standardize = FALSE,
                         intercept = FALSE)
    gfit <- glmnet(x, y, alpha = alpha,
                   family = "binomial",
                   standardize = FALSE,
                   intercept = FALSE)
    expect_equal(coef(sfit_sparse), coef(sfit_dense), tol = 0.01)
    expect_equal(coef(sfit_sparse), coef(gfit), tol = 0.01)
  }
})

test_that("regularization path is correctly computed", {
  library(glmnet)
  glmnet.control(fdev = 0)

  set.seed(1)

  sd2 <- function(x) sqrt(sum((x - mean(x))^2)/length(x))

  x <- as.matrix(with(Puromycin, cbind(conc, rate)))
  xs <- scale(x, scale = apply(x, 2, sd2))
  y <- Puromycin$state
  yy <- as.double(y)
  yyy <- ifelse(yy == 1, 0, 1)

  lambda_max <- max(abs(crossprod(yy, xs)))/nrow(x)

  fit1 <- sgdnet(x, y, family = "binomial")
  fit2 <- glmnet(x, y, family = "binomial")

  expect_equal(lambda_max, max(fit1$lambda), max(fit2$lambda))
  expect_equal(fit1$lambda, fit2$lambda)

  # Check that first solution is completely sparse
  expect_equivalent(fit1$beta[, 1], rep(0, ncol(x)))

  # Check that last solution is not
  expect_equivalent(fit1$beta[, ncol(fit1$beta)] != 0, c(TRUE, TRUE))
})

test_that("non-penalized logistic regression has similar results as glm()", {
  set.seed(1)

  x <- as.matrix(with(Puromycin, cbind(conc, rate)))
  y <- Puromycin$state

  sgdfit <- sgdnet(x, y, family = "binomial", lambda = 0, thresh = 1e-9)
  glmfit <- glm(y ~ x, family = "binomial")

  expect_equivalent(as.vector(coef(sgdfit)), coef(glmfit), tolerance = 1e-5)
})
