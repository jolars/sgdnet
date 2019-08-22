test_that("printing works for univariate families", {
  fit <- sgdnet(rnorm(10), rnorm(10))
  expect_silent(dont_print(fit))
})

test_that("printing works for multivariate families", {
  fit <- sgdnet(rnorm(100), rbinom(100, 3, 0.3), family = "multinomial")
  expect_silent(dont_print(fit))
})

test_that("printing cv_sgdnet objects works", {
  cv_fit <- cv_sgdnet(rnorm(10), rnorm(10))
  expect_silent(dont_print(cv_fit))
})
