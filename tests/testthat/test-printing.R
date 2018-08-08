context("printing")

# Suppress printning
dont_print <- function(x, ...) {
  utils::capture.output(y <- print(x, ...))
  invisible(y)
}

test_that("printing works for univariate families", {
  fit <- sgdnet(rnorm(10), rnorm(10))
  expect_silent(dont_print(fit))
})

test_that("printing works for multivariate families", {
  fit <- sgdnet(rnorm(100), rbinom(100, 3, 0.3), family = "multinomial")
  expect_silent(dont_print(fit))
})
