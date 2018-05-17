context("main model fitter")

test_that("FitModel returns correct output", {
  fit <- FitModel()
  expect_is(fit, "list")
})
