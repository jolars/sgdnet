context("gaussian regression")

test_that("gaussian regression with dense features work", {
  x <- matrix(rnorm(30), 10, 3)
  y <- as.matrix(rnorm(nrow(x)))

  fit <- sgdnet(x, y)
  expect_s3_class(fit, "sgdnet")
})


test_that("gaussian regression with sparse features work", {
  set.seed(1)
  x <- Matrix::rsparsematrix(10, 3, density = 0.5)
  y <- as.matrix(rnorm(nrow(x)))

  fit <- sgdnet(x, y, alpha = 0)

  expect_s3_class(fit, "sgdnet")
})
