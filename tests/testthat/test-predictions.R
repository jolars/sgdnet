context("predictions")

test_that("prediction for gaussian models peform as expected", {
  set.seed(1)

  library(glmnet)
  glmnet.control(fdev = 0)

  x <- with(trees, cbind(Girth, Volume))
  y <- trees$Height

  fit <- sgdnet(x, y)

  # find predictions in various ways
  pred_manual <- as.matrix(cbind(1, x) %*% coef(fit))
  pred_link <- predict(fit, x, type = "link")
  pred_response <- predict(fit, x, type = "response")

  # test that these predictions are equal
  expect_equal(pred_link, pred_response, pred_manual)

  # compare against glmnet
  fit2 <- glmnet(x, y)

  for (type in c("link", "response")) {
    expect_equal(predict(fit, x, type = type),
                 predict(fit2, x, type = type),
                 tolerance = 0.001)
  }

  # check that we get the expected nonzero indices
  pred_nonzero <- predict(fit, x, type = "nonzero")
  expect_is(pred_nonzero, "list")
  expect_length(pred_nonzero, ncol(coef(fit)))
  expect_equivalent(pred_nonzero[[1]], NULL)
  expect_equal(predict(fit, x, s = max(fit$lambda), type = "nonzero")[[1]],
               NULL)
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
})

test_that("predictions for binomial model works appropriately", {
  set.seed(1)

  library(glmnet)

  glmnet.control(fdev = 0)

  x <- as.matrix(with(infert, cbind(age, parity)))
  y <- infert$case

  sgdfit <- sgdnet(x, y, family = "binomial")
  glmfit <- glmnet(x, y, family = "binomial")

  # expect equivalent output for all the types of predictions
  for (type in c("link", "response", "class")) {
    expect_equal(predict(sgdfit, x, type = type),
                 predict(glmfit, x, type = type),
                 tolerance = 0.001)
  }
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
