context("binomial regression")

test_that("non-penalized logistic regression has similar results as glm()", {
  set.seed(1)

  x <- as.matrix(with(Puromycin, cbind(conc, rate)))
  y <- Puromycin$state

  # with intercept
  sgdfit <- sgdnet(x, y, family = "binomial", lambda = 0, thresh = 1e-9)
  glmfit <- glm(y ~ x, family = "binomial")

  expect_equivalent(as.vector(coef(sgdfit)), coef(glmfit), tolerance = 1e-5)
})

test_that("predictions for binomial model compare with glmnet", {
  set.seed(1)

  library(glmnet)

  glmnet.control(fdev = 0)

  x <- as.matrix(with(infert, cbind(age, parity)))
  y <- infert$case

  sgdfit <- sgdnet(x, y, family = "binomial")
  glmfit <- glmnet(x, y, family = "binomial")

  # expect equivalent output for all the types of predictions
  for (type in c("link", "response", "class")) {
    spred <- predict(sgdfit, x, type = type)
    gpred <- predict(glmfit, x, type = type)
    expect_equal(spred, gpred, tolerance = 0.001)
  }
})
