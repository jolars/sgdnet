context("cross-validation")

test_that("cross-validation works, including its plot and predict methods", {
  n <- 500
  p <- 2
  set.seed(2)

  alphas <- list(0, 1, c(0.2, 0.5))
  
  #
  for (family in families()) {
    for (alpha in alphas) {
      
      #
      d <- random_data(n, p, family, density = 1)
      x <- as.matrix(d$x)
      y <- d$y

      f <- getAnywhere(paste0("score.sgdnet_", family))
      types <- unlist(as.list(formals(f[["objs"]][[1]])$type.measure)[-1])

      for (type in types) {
        fit <- cv_sgdnet(x,
                         y,
                         family = family,
                         alpha = alpha,
                         nlambda = 5,
                         maxit = 10,
                         thresh = 1e-2,
                         type.measure = type)
        expect_is(fit, "cv_sgdnet")
        expect_silent(plot <- dont_plot(fit))
        expect_is(plot, "trellis")
        expect_silent(predict(fit, x))
      }
    }
  }
})

test_that("various cross-validation arguments can be used", {
  n <- 100
  x <- rnorm(100)
  y <- rnorm(100)

  # leave-one-out cross validation
  expect_silent(cv_sgdnet(x, y, nfolds = n))
  expect_silent(cv_sgdnet(x, y, foldid = sample(rep(1:10, 10))))
})
