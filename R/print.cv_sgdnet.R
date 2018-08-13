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

#' Print Method for `'cv_sgdnet'` objects
#'
#' This function prints the cross-validation summary for the `alpha` with
#' the best validation performance as well as the `lambda` at which this
#' performance was reached and the largest `lambda` (`lambda_1se`) with a
#' performance within one standard deviation of that.
#'
#' @param x an object of class `"cv_sgdnet"` as generated from a call to
#'   [cv_sgdnet()]
#' @param ... arguments passed on to [print()]
#'
#' @return Prints a [data.frame] with columns
#' \item{`type`}{type of validation performance metric}
#' \item{`alpha`}{elastic net mixing parameter}
#' \item{`lambda`}{regularization strength}
#' \item{`mean`}{mean of the prediction metric across the folds}
#' \item{`sd`}{standard deviation of the prediction metric across the folds}
#' \item{`ci_lo`}{mean minus one standard deviation}
#' \item{`ci_up`}{mean plus one standard deviation}
#'
#' @seealso [print.data.frame()], [print()], [cv_sgdnet()]
#'
#' @export
#'
#' @examples
#' fit <- cv_sgdnet(mtcars$drat, mtcars$hp)
#' print(fit)
print.cv_sgdnet <- function(x, ...) {
  ind_min <- match(x$lambda_min, x$cv_summary$lambda)
  ind_1se <- match(x$lambda_1se, x$cv_summary$lambda)

  out <- x$cv_summary[c(ind_min, ind_1se), , drop = FALSE]
  out <- cbind(type = x$name, out)
  rownames(out) <- c("lambda_min", "lambda_1se")

  print(out, ...)
  invisible(out)
}
