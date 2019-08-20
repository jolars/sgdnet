# sgdnet: Penalized Generalized Linear Models with Stochastic Gradient Descent
# Copyright (C) 2018  Johan Larsson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

#' Score the Performance of a Model fit with [sgdnet()]
#'
#' The main purpose of this function is internal use within [cv_sgdnet()],
#' where it is used to score the performance over folds in cross-validation.
#' It can, however, be used on its own to measure performance against a
#' validation set, for instance by training the model via [cv_sgdnet()]
#' and holding out a validation set for
#' use with this function.
#'
#' @param fit the model fit
#' @param x a feature matrix of new data
#' @param y response(s) for new data
#' @param type.measure the type of measure
#' @param s lambda
#' @param ... arguments passed on to [predict.sgdnet()]
#'
#' @return Returns the prediction error along the lambda path.
#'
#' @export
#'
#' @seealso [predict.sgdnet()], [cv_sgdnet()]
#'
#' @examples
#' set.seed(1)
#' n <- nrow(wine$x)
#' train_ind <- sample(n, floor(0.8*n))
#' cv_fit <- cv_sgdnet(wine$x[train_ind, ],
#'                     wine$y[train_ind],
#'                     family = "multinomial",
#'                     nfolds = 5,
#'                     alpha = c(0.5, 1))
#' score(cv_fit, wine$x[-train_ind, ], wine$y[-train_ind], "deviance")
score <- function(fit, ...) {
  UseMethod("score", fit)
}

#' @rdname score
#' @export
score.sgdnet_gaussian <- function(fit,
                                  x,
                                  y,
                                  type.measure = c("deviance", "mse", "mae"),
                                  s = fit$lambda,
                                  ...) {
  type.measure <- match.arg(type.measure)

  y <- as.vector(y)
  y_hat <- stats::predict(fit, x, s)

  switch(type.measure,
         deviance = colMeans((y_hat - y)^2),
         mse = colMeans((y_hat - y)^2),
         mae = colMeans(abs(y_hat - y)))
}

#' @rdname score
#' @export
score.sgdnet_binomial <- function(fit,
                                  x,
                                  y,
                                  type.measure = c("deviance",
                                                   "mse",
                                                   "mae",
                                                   "class",
                                                   "auc"),
                                  s = fit$lambda,
                                  ...) {
  type.measure <- match.arg(type.measure)

  prob_min <- 1e-05
  prob_max <- 1 - prob_min

  y <- as.factor(y)
  y <- diag(2)[as.numeric(y), ]

  y_hat <- stats::predict(fit, x, s = s, type = "response", ...)

  n_samples <- nrow(x)

  switch(
    type.measure,
    auc = {
      apply(y_hat, 2, function(y_hat_i) auc(y, y_hat_i))
    },
    mse = colMeans((y_hat + y[, 1] - 1)^2 + (y_hat - y[, 2])^2),
    mae = colMeans(abs(y_hat + y[, 1] - 1) + abs(y_hat - y[, 2])),
    deviance = {
      y_hat <- pmin(pmax(y_hat, prob_min), prob_max)
      lp <- y[, 1] * log(1 - y_hat) + y[, 2] * log(y_hat)
      ly <- log(y)
      ly[y == 0] <- 0
      ly <- drop((y * ly) %*% c(1, 1))
      colMeans(2 * (ly - lp))
    },
    class = colMeans(y[, 1] * (y_hat > 0.5) + y[, 2] * (y_hat <= 0.5))
  )
}

#' @rdname score
#' @export
score.sgdnet_multinomial <- function(fit,
                                     x,
                                     y,
                                     type.measure = c("deviance",
                                                      "mse",
                                                      "mae",
                                                      "class"),
                                     s = fit$lambda,
                                     ...) {
  type.measure <- match.arg(type.measure)

  prob_min <- 1e-05
  prob_max <- 1 - prob_min

  n_samples <- nrow(x)

  y <- as.factor(y)
  n_classes <- length(unique(y))
  y <- diag(n_classes)[as.numeric(y), ]

  y_hat <- stats::predict(fit, x, s = s, type = "response", ...)

  y <- array(y, dim(y_hat))

  switch(
    type.measure,
    mse = colMeans(apply((y - y_hat)^2, 3, rowSums)),
    mae = colMeans(apply(abs(y - y_hat), 3, rowSums)),
    deviance = {
      y_hat <- pmin(pmax(y_hat, prob_min), prob_max)
      lp <- y * log(y_hat)
      ly <- y * log(y)
      ly[y == 0] <- 0
      colMeans(apply(2 * (ly - lp), 3, rowSums))
    },
    class = {
      classid <- as.numeric(as.factor(apply(y_hat, 3, softmax)))
      yperm = matrix(aperm(y, c(1, 3, 2)), ncol = n_classes)
      colMeans(matrix(1 - yperm[cbind(seq(classid), classid)],
                      ncol = length(s)))
    }
  )
}

#' @rdname score
#' @export
score.sgdnet_mgaussian <- function(fit,
                                   x,
                                   y,
                                   type.measure = c("deviance", "mse", "mae"),
                                   s = fit$lambda,
                                   ...) {
  type.measure <- match.arg(type.measure)

  y_hat <- stats::predict(fit, x, s = s, ...)
  y <- array(y, dim(y_hat))

  switch(type.measure,
         deviance = colMeans(apply((y_hat - y)^2, 3, colSums)),
         mse = colMeans(apply((y_hat - y)^2, 3, colSums)),
         mae = colMeans(apply(abs(y_hat - y), 3, colSums)))
}

#' @rdname score
#' @export
score.sgdnet_poisson <- function(fit,
                                 x,
                                 y,
                                 type.measure = c("deviance", "mse", "mae"),
                                 s = fit$lambda,
                                 ...) {
  type.measure <- match.arg(type.measure)
  y <- as.vector(y)

  y_hat <- stats::predict(fit, x, s = s)

  devhat <- y * y_hat - exp(y_hat)
  devy <- y * log(y) - y
  devy[y == 0] = 0

  switch(type.measure,
         deviance = colMeans(2*(devy - devhat)),
         mse = colMeans((exp(y_hat) - y)^2),
         mae = colMeans(abs(exp(y_hat) - y)))
}

#' @rdname score
#' @export
score.cv_sgdnet <- function(fit,
                            x,
                            y,
                            type.measure,
                            s = c("lambda_1se", "lambda_min"),
                            ...) {
  s <- match.arg(s)

  score(fit$fit, x, y, type.measure, s = fit[[s]], ...)
}

#' Area Under the Curve
#'
#' @param y a matrix of responses
#' @param prob a vector of probabilities
#' @param weights weights for y
#'
#' @return Area under the ROC
#'
#' @noRd
#' @keywords internal
auc <- function(y, prob, weights = rep.int(1, nrow(y))) {
  if (is.matrix(y) || is.data.frame(y)) {

    ny <- nrow(y)
    auc(rep(c(0, 1), c(ny, ny)), c(prob,prob), as.vector(weights*y))

  } else {

    if (is.null(weights)) {
      rprob <- rank(prob)
      n1 <- sum(y)
      n0 <- length(y) - n1
      u <- sum(rprob[y == 1]) - n1*(n1 + 1)/2
      exp(log(u) - log(n1) - log(n0))
    } else {
      # randomize ties
      rprob <- stats::runif(length(prob))
      op <- order(prob, rprob)
      y <- y[op]
      weights <- weights[op]
      cw <- cumsum(weights)
      w1 <- weights[y == 1]
      cw1 <- cumsum(w1)
      wauc <- log(sum(w1 * (cw[y == 1] - cw1)))
      sumw1 <- cw1[length(cw1)]
      sumw2 <- cw[length(cw)] - sumw1
      exp(wauc - log(sumw1) - log(sumw2))
    }
  }
}
