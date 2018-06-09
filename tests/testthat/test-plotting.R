context("plotting")

test_that("we can produce plots for all out models", {
  dont_plot <- function(x, ...) {
    tmp <- tempfile()
    grDevices::png(tmp)
    p <- graphics::plot(x, ...)
    grDevices::dev.off()
    unlink(tmp)
    invisible(p)
  }

  y <- mtcars$mpg
  x <- subset(mtcars, select = -mpg)

  fit <- sgdnet(x, y)

  expect_is(dont_plot(fit, "norm"), "trellis")
  expect_is(dont_plot(fit, "lambda"), "trellis")
})
