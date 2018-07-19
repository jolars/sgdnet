context("assertions")

test_that("wrong input to predict.sgdnet() return errors", {
  x <- with(trees, cbind(Girth, Volume))
  y <- trees$Height

  fit <- sgdnet(x, y)

  expect_error(predict(fit))
  expect_error(predict(fit, x, type = 1))
  expect_error(predict(fit, x, s = -3))
  expect_error(predict(fit, x, type = "class"))

  fit <- sgdnet(iris[, 1:2], iris[, 5], family = "multinomial")
  expect_error(predict(fit, x, s = -3, type = "class"))
})

test_that("wrong input to sgdnet() returns errors", {
  x <- with(trees, cbind(Girth, Volume))
  y <- trees$Height
  y_na <- y
  y_na[1] <- NA

  expect_error(sgdnet(x, y, lambda = double(0)))
  expect_error(sgdnet(x, y, maxit = 0))
  expect_error(sgdnet(x, as.factor(y)))
  expect_error(sgdnet(x, y, thresh = -0.1))
  expect_error(sgdnet(x, y_na))
  expect_error(sgdnet(x, cbind(y, y)))
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
  y <- rep.int(0, nrow(x))
  expect_error(sgdnet(x, y, family = "binomial"))
  y <- c(0, 0, 1)
  x <- c(0.5, 0.5, 0.5)
  expect_error(sgdnet(x, y, family = "binomial"))
})
