# compare linear interpolation against new predictions
library(sgdnet)

fit <- sgdnet(trees$Girth, trees$Volume)
pred_exact <- predict(fit,
                      trees$Girth,
                      s = 0.001,
                      exact = TRUE,
                      x = trees$Girt,
                      y = trees$Volume)
pred_approx <- predict(fit, trees$Girth, s = 0.001)

if (!isTRUE(all.equal(pred_exact, pred_approx, tolerance = 1e-9))) {
  stop("linear interpolation does approximate the exact solution")
}
