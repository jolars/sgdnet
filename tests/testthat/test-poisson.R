test_that("solutions along regularization path are the same as glmnet", {
  set.seed(1)

  N <- 2000
  p <- 10
  nzc <- 2
  x <- matrix(rnorm(N*p),N,p)
  beta <- rnorm(nzc)
  f <- x[,seq(nzc)]%*%beta
  mu <- exp(f)
  y <- rpois(N,mu)
  
  sgd_l1 <- sgdnet(x,y,family="poisson",alpha=1,nlambda=10,thresh=1e-9)
  glm_l1 <- glmnet(x,y,family="poisson",alpha=1,nlambda=10,thresh=1e-9)
  
  sgd_l2 <- sgdnet(x,y,family="poisson",alpha=0,nlambda=10,thresh=1e-9)
  glm_l2 <- glmnet(x,y,family="poisson",alpha=0,nlambda=10,thresh=1e-9)

  expect_equivalent(sgd_l2$nulldev, glm_l2$nulldev, tolerance = 1e-10)
  expect_equivalent(sgd_l1$nulldev, glm_l1$nulldev, tolerance = 1e-10)
  
  expect_equivalent(coef(sgd_l2), coef(glm_l2), tolerance = 1e-3)
  expect_equivalent(coef(sgd_l1), coef(glm_l1), tolerance = 1e-3)

})
