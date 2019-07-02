context("multivariate gaussian regression")

test_that("we can approximate the closed form multivariate ridge regression solution", {
  set.seed(1)

  n <- 500
  p <- 3
  b <- cbind(c(-5, 3, 2), c(0, -5, 9))

  x <- scale(matrix(rnorm(p * n), nrow = n))
  E <- x %*% b
  y <- cbind(rnorm(n, colMeans(E)[1]), rnorm(n, colMeans(E)[2]))
  lambda <- 0.01
  sd_y <- apply(y, 2, function(y) sqrt(var(y)*(n - 1) / n))

  beta_theoretical <- solve((crossprod(x) + lambda*diag(p))) %*% crossprod(x, y)
  sgdnet_fit <- sgdnet(x, y,
                       family = "mgaussian",
                       alpha = 0,
                       lambda = sd_y[1]*lambda/n,
                       intercept = FALSE,
                       thresh = 1e-5,
                       maxit = 1e4)
  
  batch_fit <- sgdnet(x, y,
                       family = "mgaussian",
                       alpha = 0,
                       lambda = sd_y[1]*lambda/n,
                       intercept = FALSE,
                       thresh = 1e-5,
                       maxit = 1e4,
                       batchsize = 10)

  expect_equivalent(beta_theoretical,
                    as.matrix(do.call(cbind, coef(sgdnet_fit))[-1, ]),
                    tolerance = 1e-6)
})

test_that("standardizing responses works", {
  library(glmnet)
  glmnet.control(fdev = 0)

  y <- student$y
  x <- student$x
  n <- nrow(x)
  sd2 <- function(x) sqrt(sum((x - mean(x))^2)/length(x))

  y_standardized <- scale(y, scale = apply(y, 2, sd2))

  sfit <- sgdnet(x, y, family = "mgaussian", standardize.response = TRUE)
  bfit <- sgdnet(x, y, family = "mgaussian", standardize.response = TRUE, batchsize = 10)
  gfit <- glmnet(x, y, family = "mgaussian", standardize.response = TRUE)

  expect_equal(sfit$lambda, gfit$lambda)
  expect_equivalent(coef(sfit), coef(gfit))
  expect_equivalent(coef(bfit), coef(gfit), tolerance = 1e-2)
})
