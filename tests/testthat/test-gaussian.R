context("gaussian regression")

test_that("FitModel returns correct output", {
  x <- matrix(rnorm(30), 10, 3)
  y <- as.matrix(rnorm(nrow(x)))

  fit <- sgdnet(x, y)
  expect_s3_class(fit, "sgdnet")
})
