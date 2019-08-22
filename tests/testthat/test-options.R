test_that("sgdnet options can be set and respected", {
  opts <- options()
  options(sgdnet.debug = TRUE)

  x <- cars$speed
  y <- cars$dist

  fit <- sgdnet(x, y)

  diagnostics <- attr(fit, "diagnostics")
  loss <- diagnostics$loss
  expect_is(loss, "list")
  expect_true(all(unlist(lapply(loss, function(x) x > 0 & x < Inf))))

  options(opts)
})
