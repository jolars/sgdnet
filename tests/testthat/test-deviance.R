context("deviance")

test_that("we receive the correct deviance from deviance.sgdnet()", {
  set.seed(1)

  library(glmnet)
  glmnet.control(fdev = 0)

  x <- with(rock, cbind(area, peri, shape))
  y <- rock$perm

  sfit1 <- sgdnet(x, y, thresh = 1e-10)
  gfit1 <- glmnet(x, y, thresh = 1e-10)

  expect_equal(deviance(sfit1), deviance(gfit1), tolerance = 0.001)

  sfit2 <- update(sfit1, alpha = 0)
  gfit2 <- update(gfit1, alpha = 0)

  expect_equal(deviance(sfit2), deviance(gfit2), tolerance = 0.001)
  expect_equal(deviance(sfit1), (1 - sfit1$dev.ratio)*sfit1$nulldev)
  expect_equal(sfit1$dev.ratio[1], 0)
  expect_equal(sfit1$nulldev, sum((y - mean(y))^2))
})
