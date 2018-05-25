context("plotting")

test_that("we can produce plots for all out models", {
  y <- mtcars$mpg
  x <- subset(mtcars, select = -mpg)

  fit <- sgdnet(x, y)

  plot <- plot(fit)

  expect_is(plot, "trellis")
})
