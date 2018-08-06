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
  set.seed(2)

  n <- 100
  p <- 2

  x <- matrix(rnorm(n*p), n, p)

  grid <- expand.grid(
    family = c("gaussian", "binomial", "multinomial", "mgaussian"),
    exact = c(TRUE, FALSE),
    s = c(0, 1/n),
    type = c("link", "response", "coefficients", "nonzero", "class"),
    stringsAsFactors = FALSE
  )

  for (i in seq_len(nrow(grid))) {
    family <- grid$family[i]
    type   <- grid$type[i]
    exact  <- grid$exact[i]
    s      <- grid$s[i]

    if (type == "class" && !(family %in% c("binomial", "multinomial")))
      next

    y <- switch(family,
                gaussian = rnorm(n, 10, 2),
                binomial = rbinom(n, 1, 0.8),
                multinomial = rbinom(n, 3, 0.5),
                mgaussian = cbind(rnorm(n, -10), rnorm(n, 10)))

    fit <- sgdnet(x, y, family = family, maxit = 10, thresh = 1e-1)

    args <- list(object = fit,
                 newx = x,
                 s = s,
                 exact = exact,
                 x = x,
                 y = y,
                 family = family)

    expect_silent(do.call(predict, args))
  }
})

test_that("linear interpolation succeeds", {
  x <- with(trees, cbind(Girth, Volume))
  y <- trees$Height

  fit <- sgdnet(x, y)

  pred_old <- predict(fit, x, s = 0.04, type = "coefficients")
  pred_new <- predict(sgdnet(x, y, lambda = 0.04), type = "coefficients")

  expect_equivalent(pred_old, pred_new, tolerance = 0.01)
  expect_is(pred_new, "dgCMatrix")

  pred_one <- predict(fit, x, s = 0.03, type = "coefficients")
  pred_two <- predict(fit, x, s = 0.05, type = "coefficients")

  expect_true(all(abs(as.vector(pred_one)) > abs(as.vector(pred_old))))
  expect_true(all(abs(as.vector(pred_two)) < abs(as.vector(pred_old))))
})

test_that("refitting works when exact = TRUE", {
  set.seed(1)

  fit <- sgdnet(trees$Girth, trees$Volume)
  pred_exact <- predict(fit,
                        trees$Girth,
                        s = 0.001,
                        exact = TRUE,
                        x = trees$Girt,
                        y = trees$Volume)
  pred_approx <- predict(fit, trees$Girth, s = 0.001)

  expect_equal(pred_exact, pred_approx, tolerance = 1e-4)
})

test_that("NAs in new data are handled appropriately", {
  set.seed(1)

  # For univariate case
  x <- trees$Girth
  y <- trees$Volume

  fit <- sgdnet(x, y)
  x[1] <- NA
  expect_silent(predict(fit, x))

  # For multivariate case
  x <- cbind(mtcars$hp, mtcars$drat)
  y <- cbind(mtcars$disp, mtcars$vs)

  fit <- sgdnet(x, y, family = "mgaussian")
  x[10, ] <- NA
  expect_silent(predict(fit, x))
})


