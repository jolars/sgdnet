context("options")

test_that("sgdnet options can be set and respected", {
  opts <- options()
  options(sgdnet.debug = TRUE)

  x <- cars$speed
  y <- cars$dist

  fit <- sgdnet(x, y)

  diagnostics <- attr(fit, "diagnostics")
  loss <- diagnostics$loss
  expect_is(loss, "list")

  options(opts)
})
