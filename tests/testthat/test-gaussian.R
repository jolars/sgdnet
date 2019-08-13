context("gaussian regression")

test_that("we can approximately reproduce the OLS solution", {
  set.seed(1)

  airquality <- na.omit(airquality)
  x <- as.matrix(subset(airquality, select = c("Wind", "Temp")))
  y <- airquality$Ozone

  sgd_fit <- sgdnet(x, y, lambda = 0, maxit = 1000, thresh = 0.0001)
  sgd_fit_batch <- sgdnet(x, y, lambda = 0, maxit = 1000, thresh = 0.00001, batchsize = 10)
  sgd_fit_cyclic <- sgdnet(x, y, lambda = 0, maxit = 1000, thresh = 0.000001, cyclic = TRUE)
  ols_fit <- lm(y ~ x)

  expect_equivalent(coef(ols_fit), as.vector(coef(sgd_fit)),
                    tolerance = 0.001)
  expect_equivalent(coef(ols_fit), as.vector(coef(sgd_fit_batch)),
                    tolerance = 0.001)
  expect_equivalent(coef(ols_fit), as.vector(coef(sgd_fit_cyclic)),
                    tolerance = 0.001)
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
  fit_batch <- sgdnet(x, y, maxit = 1000, thresh = 0.0001, batchsize = 10)
  fit_cyclic <- sgdnet(x, y, maxit = 1000, thresh = 0.0001, cyclic = TRUE)

  expect_equal(max(fit$lambda), lambda_max)
  expect_equivalent(as.matrix(fit$beta[, 1]), cbind(rep(0, 3)))
  expect_equal(max(fit_batch$lambda), lambda_max)
  expect_equivalent(as.matrix(fit_batch$beta[, 1]), cbind(rep(0, 3)))
  expect_equal(max(fit_cyclic$lambda), lambda_max)
  expect_equivalent(as.matrix(fit_cyclic$beta[, 1]), cbind(rep(0, 3)))
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
  
  batch_fit <- sgdnet(x, y,
                       alpha = 0,
                       lambda = sd_y*lambda/n,
                       intercept = FALSE,
                       thresh = 0.00001,
                       maxit = 1000,
                       batchsize = 10)

  batch_full <- sgdnet(x, y,
                      alpha = 0,
                      lambda = sd_y*lambda/n,
                      intercept = FALSE,
                      thresh = 0.000001,
                      maxit = 1000,
                      batchsize = 500)

  expect_equivalent(beta_theoretical, coef(sgdnet_fit)[-1],
                    tolerance = 1e-3)
  expect_equivalent(beta_theoretical, coef(batch_fit)[-1],
                    tolerance = 1e-3)
  expect_equivalent(beta_theoretical, coef(batch_full)[-1],
                    tolerance = 1e-3)
})

test_that("a constant response returns a completely sparse solution with intercept at mean(y)", {
  x <- as.matrix(iris[, 1:4])
  y <- rep(5, nrow(x))

  sfit <- sgdnet(x, y)

  expect_equal(sfit$lambda, rep(0, length(sfit$lambda)))
  expect_equal(as.vector(sfit$beta), rep(0, length(sfit$beta)))
  expect_equal(as.vector(sfit$a0), rep(mean(y), length(sfit$a0)))
})
