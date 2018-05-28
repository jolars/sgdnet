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
  fit_sparse <- sgdnet(x, y, alpha = 0, intercept = FALSE)
  set.seed(seed)
  fit_dense <- sgdnet(as.matrix(x), y, alpha = 0, intercept = FALSE)

  expect_equal(coef(fit_sparse), coef(fit_dense))
})

test_that("we can approximately reproduce the OLS solution", {
  airquality <- na.omit(airquality)
  x <- as.matrix(subset(airquality, select = c("Wind", "Temp")))
  y <- airquality$Ozone

  sgd_fit <- sgdnet(x, y, lambda = 0, maxit = 1000, thresh = 0.0001)
  ols_fit <- lm(y ~ x)

  expect_equivalent(coef(ols_fit), as.vector(coef(sgd_fit)),
                    tolerance = 0.01)
})

test_that("all weights are zero when lambda > lambda_max", {
  # Code for this test has been borrowed from
  # https://stats.stackexchange.com/questions/166630/glmnet-compute-maximal-lambda-value

  # TODO(jolars): current this test is useless since the lasso implementation
  #               is defunct

  set.seed(1)

  n <- 500
  p <- 3
  b <- c(-5, 3, 2)

  x <- scale(matrix(rnorm(p * n), nrow = n))
  y <- rnorm(n, mean = x%*%b)

  alpha <- 1

  lambda_max <- max(abs(t(y - mean(y)*(1 - mean(y))) %*% x)) / (alpha*n)

  fit <- sgdnet(x, y, alpha = 1, lambda = lambda_max, maxit = 1000,
                thresh = 0.0001)

  # fit <- sgdnet(x, y, alpha = 1, lambda = lambda_max *0.9, maxit = 1000,
  #               thresh = 0.0001)

  expect_equal(coef(fit)[-1], rep(0, length(coef(fit)) - 1))
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
