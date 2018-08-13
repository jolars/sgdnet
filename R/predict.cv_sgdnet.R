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

#' Make Predcitions Based on Fit From [cv_sgdnet()]
#'
#' This function is mostly for convenience, since it makes it easy to
#' take an object from a call to [cv_sgdnet()] and make predictions based
#' on a \eqn{\lambda}{lambda} chosen from cross-validation results.
#'
#' @param object a fit from [cv_sgdnet()]
#' @param newx new data to base predictions on
#' @param s `'lambda.1se'` chooses predictions based on the
#'   model fit to the largest \eqn{\lambda}{lambda} with an error
#'   at most one standard deviation away from the fit with the least
#'   error; predictions are based on the latter fit if `'lambda.min'` is
#'   chosen
#' @param ... arguments passed on to [predict.sgdnet()]
#'
#' @inherit predict.sgdnet return
#' @export
#'
#' @examples
#' set.seed(1)
#' train_ind <- sample(150, 100)
#' fit <- cv_sgdnet(iris[train_ind, 1:4],
#'                  iris[train_ind, 5],
#'                  family = "multinomial",
#'                  nfolds = 5)
#' predict(fit, iris[-train_ind, 1:4], s = "lambda_min", type = "class")
predict.cv_sgdnet <- function(object,
                              newx,
                              s = c("lambda_1se", "lambda_min"),
                              ...) {
  if (is.character(s)) {
    s <- match.arg(s)
    s <- object[[s]]
  }
  stats::predict(object$fit, newx, s = s, ...)
}
