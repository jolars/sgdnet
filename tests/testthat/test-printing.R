context("printing")

test_that("printing works for univariate families", {
  set.seed(0)
  fit <- sgdnet(rnorm(10), rnorm(10))
  expect_silent(dont_print(fit))
})

test_that("printing works for multivariate families", {
  set.seed(2)
  fit <- sgdnet(rnorm(100), rbinom(100, 3, 0.3), family = "multinomial")
  expect_silent(dont_print(fit))
})

test_that("printing cv_sgdnet objects works", {
  set.seed(0)
  cv_fit <- cv_sgdnet(rnorm(10), rnorm(10))
  expect_silent(dont_print(cv_fit))
})
