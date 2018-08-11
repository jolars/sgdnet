context("plotting")

test_that("we can produce plots for all out models", {
  y <- mtcars$mpg
  x <- subset(mtcars, select = -mpg)

  fit <- sgdnet(x, y)

  expect_is(dont_plot(fit, "norm"), "trellis")
  expect_is(dont_plot(fit, "lambda"), "trellis")
  expect_is(dont_plot(fit, "dev"), "trellis")

  y <- mtcars$am
  x <- mtcars$mpg

  fit <- sgdnet(x, y, family = "binomial")

  expect_is(dont_plot(fit, "norm"), "trellis")
  expect_is(dont_plot(fit, "lambda"), "trellis")
  expect_is(dont_plot(fit, "dev"), "trellis")

  y <- mtcars$gear
  x <- mtcars$mpg

  fit <- sgdnet(x, y, family = "multinomial")

  expect_is(dont_plot(fit, "norm"), "trellis")
  expect_is(dont_plot(fit, "lambda"), "trellis")
  expect_is(dont_plot(fit, "dev"), "trellis")

  y <- cbind(mtcars$hp, mtcars$drat)
  x <- mtcars$disp

  fit <- sgdnet(x, y, family = "mgaussian")

  expect_is(dont_plot(fit, "norm"), "trellis")
  expect_is(dont_plot(fit, "lambda"), "trellis")
  expect_is(dont_plot(fit, "dev"), "trellis")
})
