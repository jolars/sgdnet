context("input")

test_that("erroneous input throws errors", {
  x <- matrix(1:9, 3, 3)
  y <- rnorm(3)

  # Input to all families
  expect_error(sgdnet(x, rep(NA, 3)))
  expect_error(sgdnet(x, y, alpha = -1))
  expect_error(sgdnet(x, y, lambda = -1))
  expect_error(sgdnet(x, y, intercept = 3))
  expect_error(sgdnet(x, y, family = "asdf"))
  expect_error(sgdnet(x, c(y, 1)))
  expect_error(sgdnet(matrix(NA, 3, 3), y, family = "gaussian"))

  # Input to gaussian regression
  expect_error(sgdnet(x, as.factor(y), family = "gaussian"))
  expect_error(sgdnet(x, cbind(y, y), family = "gaussian"))
})
