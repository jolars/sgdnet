context("binomial regression")

test_that("basic model fitting with dense and sparse features", {
  library(glmnet)
  glmnet.control(fdev = 0)

  set.seed(1)

  n <- 1000
  d <- 3

  sd2 <- function(x) sqrt(sum((x - mean(x))^2)/length(x))

  # Sparse features
  x <- Matrix::rsparsematrix(n, d, density = 0.5)
  x <- Matrix::Matrix(apply(x, 2, function(x) x/sd2(x)), sparse = TRUE)
  y <- rbinom(n, 1, 0.5)

  fit1 <- sgdnet(x, y, family = "binomial", standardize = FALSE,
                 intercept = TRUE)
  fit2 <- glmnet(x, y, family = "binomial", standardize = FALSE,
                 intercept = TRUE)

  expect_is(fit1, "sgdnet")
  expect_is(fit1, "sgdnet_binomial")
  expect_equivalent(coef(fit1), coef(fit2), tolerance = 0.01)

  fit3 <- sgdnet(as.matrix(x), y, family = "binomial", intercept = TRUE)
  fit4 <- glmnet(as.matrix(x), y, family = "binomial", intercept = TRUE)

  expect_equal(coef(fit3), coef(fit4), tolerance = 0.01)
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

  expect_equivalent(as.vector(coef(sgdfit)), coef(glmfit))
})
