context("gaussian regression")

test_that("gaussian regression with dense features work as expected", {
  x <- iris[, 1:3]
  y <- iris[, 4]

  # Check that we get the expected class
  sgd_fit <- sgdnet(x, y, alpha = 0)
  expect_s3_class(sgd_fit, "sgdnet")
})


test_that("gaussian regression with sparse and dense features work", {
  n <- 100
  d <- 3

  x <- Matrix::rsparsematrix(n, d, density = 0.5)
  y <- x[, 1]*runif(n, 0.1, 0.2) + x[, 2]*rnorm(n, 0.4)

  seed <- sample.int(100, 1)

  set.seed(seed)
  fit_sparse <- sgdnet(x, y, alpha = 0, intercept = FALSE, standardize = FALSE)
  set.seed(seed)
  fit_dense <- sgdnet(as.matrix(x), y, alpha = 0, intercept = FALSE,
                      standardize = FALSE)
  gfit <- sgdnet(x, y, alpha = 0, intercept = FALSE, standardize = FALSE)

  expect_equal(coef(fit_sparse), coef(fit_dense), tolerance = 1e-3)
  expect_equal(coef(gfit), coef(fit_sparse), tolerance = 1e-3)
})

test_that("we can approximately reproduce the OLS solution", {
  set.seed(1)

  airquality <- na.omit(airquality)
  x <- as.matrix(subset(airquality, select = c("Wind", "Temp")))
  y <- airquality$Ozone

  sgd_fit <- sgdnet(x, y, lambda = 0, maxit = 1000, thresh = 0.0001)
  ols_fit <- lm(y ~ x)

  expect_equivalent(coef(ols_fit), as.vector(coef(sgd_fit)),
                    tolerance = 0.01)
})

test_that("all weights are zero when lambda > lambda_max", {
  set.seed(1)

  x <- iris[, 1:3]
  y <- iris[, 4]

  sd2 <- function(x) sqrt(sum((x - mean(x))^2)/length(x))
  sy <- sd2(y)
  xx <- scale(x, scale = apply(x, 2, sd2))
  yy <- (y - mean(y))/sy

  alpha <- 1

  lambda_max <- max(abs(crossprod(yy, xx)) * sy)/NROW(x)

  fit <- sgdnet(x, y, maxit = 1000, thresh = 0.0001)

  max(fit$lambda)

  expect_equal(max(fit$lambda), lambda_max)
  expect_equivalent(as.matrix(fit$beta[, 1]), cbind(rep(0, 3)))
})

test_that("we can approximate the closed form ridge regression solution", {
  set.seed(1)

  n <- 500
  p <- 3
  b <- c(-5, 3, 2)

  x <- scale(matrix(rnorm(p * n), nrow = n))
  y <- rnorm(n, mean = x%*%b)
  lambda <- 0.01
  sd_y <- sqrt(var(y)*(n - 1) / n)

  beta_theoretical <- solve((crossprod(x) + lambda*diag(p))) %*% crossprod(x, y)
  sgdnet_fit <- sgdnet(x, y,
                       alpha = 0,
                       lambda = sd_y*lambda/n,
                       intercept = FALSE,
                       thresh = 0.00001,
                       maxit = 1000)

  expect_equivalent(beta_theoretical, coef(sgdnet_fit)[-1],
                    tolerance = 1e-3)
})

test_that("we generate the same lambda path as in glmnet", {
  set.seed(1)

  x <- with(trees, cbind(Girth, Height))
  y <- trees$Volume

  library(glmnet)
  glmnet.control(fdev = 0) # make sure that the whole lambda path is returned

  glmnetfit <- glmnet(x, y)
  sgdnetfit <- sgdnet(x, y)

  expect_equal(glmnetfit$lambda, sgdnetfit$lambda)
})


test_that("a constant response returns a completely sparse solution with intercept at mean(y)", {
  x <- as.matrix(iris[, 1:4])
  y <- rep(5, nrow(x))

  sfit <- sgdnet(x, y)

  expect_equal(sfit$lambda, rep(0, length(sfit$lambda)))
  expect_equal(as.vector(sfit$beta), rep(0, length(sfit$beta)))
  expect_equal(as.vector(sfit$a0), rep(mean(y), length(sfit$a0)))
})
