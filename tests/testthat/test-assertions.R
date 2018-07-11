context("assertions")

test_that("output types are those that we expect", {
  expect_equal(2 * 2, 4)
})

test_that("wrong input to predict.sgdnet() return errors", {
  x <- with(trees, cbind(Girth, Volume))
  y <- trees$Height

  fit <- sgdnet(x, y)

  expect_error(predict(fit))
  expect_error(predict(fit, x, type = 1))
  expect_error(predict(fit, x, s = -3))
  expect_error(predict(fit, x, type = "class"))
})

test_that("wrong input to sgdnet() returns errors", {
  x <- with(trees, cbind(Girth, Volume))
  y <- trees$Height

  expect_error(sgdnet(x, y[-1]))
  expect_error(sgdnet(x, y, family = "binomial"))
  expect_error(sgdnet(x, y, lambda = -1))
  expect_error(sgdnet(x, rep(NA, 3)))
  expect_error(sgdnet(x, y, alpha = -1))
  expect_error(sgdnet(x, y, lambda = -1))
  expect_error(sgdnet(x, y, intercept = 3))
  expect_error(sgdnet(x, y, family = "asdf"))
  expect_error(sgdnet(x, c(y, 1)))
  expect_error(sgdnet(matrix(NA, 3, 3), y, family = "gaussian"))

  y <- rbinom(nrow(x), 3, 0.5)
  expect_error(sgdnet(x, y, family = "binomial"))
  y <- rbinom(nrow(x), 1, 0.5)
  expect_error(sgdnet(x, y, family = "multinomial"))
})
