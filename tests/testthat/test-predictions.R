context("predictions")

test_that("prediction for gaussian models peform as expected", {
  set.seed(1)

  x <- with(trees, cbind(Girth, Volume))
  y <- trees$Height

  fit <- sgdnet(x, y)

  # find predictions in various ways
  pred_manual <- as.matrix(cbind(1, x) %*% coef(fit))
  pred_link <- predict(fit, x, type = "link")
  pred_response <- predict(fit, x, type = "response")

  # test that these predictions are equal
  expect_equal(pred_link, pred_response, pred_manual)

  pred_nonzero <- predict(fit, x, type = "nonzero")
  expect_is(pred_nonzero, "list")
  expect_length(pred_nonzero, ncol(coef(fit)))
  expect_equivalent(pred_nonzero[[1]], NULL)

  pred_coefficients <- predict(fit, x, type = "coefficients")
  expect_equal(pred_coefficients, coef(fit))

  # check linear interpolation
  pred_new <- predict(fit, x, s = 0.04, type = "coefficients")
  pred_two <- predict(sgdnet(x, y, lambda = 0.04), type = "coefficients")
  expect_is(pred_new, "dgCMatrix")
  expect_equivalent(pred_new, pred_two, tolerance = 0.1)
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
