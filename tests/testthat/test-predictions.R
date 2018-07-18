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
  expect_equal(pred_link, pred_response)
  expect_equal(pred_link, pred_manual)

  # check that we get the expected nonzero indices
  pred_nonzero <- predict(fit, x, type = "nonzero")
  expect_is(pred_nonzero, "list")
  expect_length(pred_nonzero, ncol(coef(fit)))
  expect_equal(predict(fit, x, s = 0, type = "nonzero")[[1]], seq_len(ncol(x)))

  pred_coefficients <- predict(fit, x, type = "coefficients")
  expect_equal(pred_coefficients, coef(fit))

  # check linear interpolation
  pred_old <- predict(fit, x, s = 0.04, type = "coefficients")
  pred_new <- predict(sgdnet(x, y, lambda = 0.04), type = "coefficients")

  expect_equivalent(pred_old, pred_new, tolerance = 0.01)
  expect_is(pred_new, "dgCMatrix")

  pred_one <- predict(fit, x, s = 0.03, type = "coefficients")
  pred_two <- predict(fit, x, s = 0.05, type = "coefficients")

  expect_true(all(abs(as.vector(pred_one)) > abs(as.vector(pred_old))))
  expect_true(all(abs(as.vector(pred_two)) < abs(as.vector(pred_old))))

  # check that we can have data.frame as new input
  fit <- sgdnet(iris[, 1:4], iris$Species, family = "multinomial")
  expect_error(predict(fit, newx = iris[, 1:4]), NA)

  # check that we can have a vector as new data
  fit <- sgdnet(mtcars$mpg, mtcars$drat)
  expect_error(predict(fit, newx = mtcars$drat), NA)
})
