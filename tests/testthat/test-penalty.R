context("penalty")

test_that("we expect lasso solution when the non-convexity parameter is large enough", {
  set.seed(2)

  airquality <- na.omit(airquality)
  x <- scale(as.matrix(subset(airquality, select = c("Wind", "Temp"))))
  y <- scale(airquality$Ozone)

  sgd_fit <- sgdnet(x, y, alpha = 1, thresh=1e-9)
  sgd_mcp <- sgdnet(x, y, non_convexity = 1e+10, lambda = sgd_fit$lambda, thresh=1e-9, penalty = "MCP")
  sgd_scad <- sgdnet(x, y, non_convexity = 1e+10, lambda = sgd_fit$lambda,thresh=1e-9,  penalty = "SCAD")

  expect_equivalent(coef(sgd_fit), coef(sgd_mcp),
                    tolerance = 1e-7)

  expect_equivalent(coef(sgd_fit), coef(sgd_scad),
                    tolerance = 1e-7)
})
