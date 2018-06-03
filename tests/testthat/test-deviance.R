context("deviance")

test_that("we receive the correct deviance from deviance.sgdnet()", {
  set.seed(1)

  library(glmnet)
  glmnet.control(fdev = 0)

  d <- 2
  n <- 1000
  x <- matrix(rnorm(n*d), nrow = n, ncol = d)

  # loop over all families
  for (family in c("gaussian", "binomial")) {
    y <- switch(family,
                gaussian = rnorm(n),
                binomial = rbinom(n, 1, 0.5))

    # return deviance of intercept-only model
    nulldev <- switch(family,
                      gaussian = sum((y - mean(y))^2),
                      binomial = -2*sum(y*log(mean(y)/(1 - mean(y)))
                                        - log(1 + exp(log(mean(y)
                                                          /(1 - mean(y)))))))

    # loop over variations of penalties (ridge, lasso, elastic net)
    for (alpha in c(0, 0.5, 1)) {
      sfit <- sgdnet(x, y, alpha = alpha)
      gfit <- glmnet(x, y, alpha = alpha)

      expect_equal(deviance(sfit), deviance(gfit), tolerance = 0.01)
      expect_equal(deviance(sfit), (1 - sfit$dev.ratio)*sfit$nulldev)
      expect_equal(sfit$nulldev, gfit$nulldev, nulldev)
    }
  }
})
