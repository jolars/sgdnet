context("predictions")

test_that("prediction for gaussian models peform as expected", {
  set.seed(1)

  x <- with(trees, cbind(Girth, Volume))
  y <- trees$Height

  fit <- sgdnet(x, y)

  pred_link <- predict(fit, x, type = "link")
  pred_response <- predict(fit, x, type = "response")

  # link and response should be the same for the gaussian model
  expect_equal(pred_link, pred_response)

  pred_nonzero <- predict(fit, x, type = "nonzero")
  expect_is(pred_nonzero, "list")

  pred_coefficients <- predict(fit, x, type = "coefficients")
  expect_equal(pred_coefficients, coef(fit))

  # check linear interpolation
  pred_new <- predict(fit, x, s = 0.04)
  expect_is(pred_new, "matrix")
})

test_that("assertations for incorrect input throw errors", {
  x <- with(trees, cbind(Girth, Volume))
  y <- trees$Height

  fit <- sgdnet(x, y)

  expect_error(predict(fit))
  expect_error(predict(fit, x, exact = "yes"))
  expect_error(predict(fit, x, type = 1))
  expect_error(predict(fit, x, s = -3))
})
