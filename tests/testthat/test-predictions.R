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
})

test_that("predictions run smoothly for a range of combinations and options", {
  x1 <- cbind(mtcars$mpg, mtcars$hp)
  x2a <- mtcars$mpg
  x2b <- x2a
  x2b[1] <- NA
  for (family in c("gaussian", "binomial", "multinomial")) {
    y <- switch(family,
                gaussian = mtcars$drat,
                binomial = mtcars$vs,
                multinomial = mtcars$gear)

    fit1a <- sgdnet(x1, y, family = family)
    fit1a <- sgdnet(x1, y, lambda = 0.0001, family = family)
    fit2a <- sgdnet(x2a, y, lambda = 0.0001, family = family)

    for (type in c("link", "response", "coefficients", "nonzero", "class")) {
      if (type == "class" && !(family %in% c("binomial", "multinomial")))
        next

      expect_error(predict(fit1a, newx = x1, type = type), NA)
      expect_error(predict(fit1a, newx = x1, s = 0.001), NA)
      expect_error(predict(fit2a, newx = x2a, type = type), NA)
      expect_error(predict(fit2a, newx = x2a, s = 0.001), NA)
      expect_error(predict(fit2b, newx = x2b))
    }
  }
})

test_that("linear interpolation succeeds", {
  pred_old <- predict(fit, x, s = 0.04, type = "coefficients")
  pred_new <- predict(sgdnet(x, y, lambda = 0.04), type = "coefficients")

  expect_equivalent(pred_old, pred_new, tolerance = 0.01)
  expect_is(pred_new, "dgCMatrix")

  pred_one <- predict(fit, x, s = 0.03, type = "coefficients")
  pred_two <- predict(fit, x, s = 0.05, type = "coefficients")

  expect_true(all(abs(as.vector(pred_one)) > abs(as.vector(pred_old))))
  expect_true(all(abs(as.vector(pred_two)) < abs(as.vector(pred_old))))
})



