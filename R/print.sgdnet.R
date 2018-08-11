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

#' Print a Path Summary for a Fit from sgdnet
#'
#' @param x an object of class `'sgdnet'`, typically the result of
#'          calling [sgdnet()]
#' @param ... other arguments passed to [print()]
#'
#' @return Prints (and return invisibly) a table of the regularization path from
#' the fit with the following columns:
#' \tabular{ll}{
#'   `Df`     \tab pseudo-degrees of freedom, namely the number of non-zero
#'                 coefficients \cr
#'   `%Dev`   \tab the percentage of deviance explained \cr
#'   `Lambda` \tab the \eqn{\lambda}{lambda} value (regularization strength) of
#'                 the fit
#' }
#' @export
#' @seealso [sgdnet()], [print.sgdnet()]
#'
#' @examples
#' fit <- sgdnet(with(mtcars, cbind(drat, hp)), mtcars$disp)
#' print(fit, digits = 1)
print.sgdnet <- function(x, ...) {
  cat("\nCall: ", deparse(x$call), "\n\n")
  out <- cbind(Df = x$df,
               "%Dev" = x$dev.ratio,
               Lambda = x$lambda)
  print(out, ...)
  invisible(out)
}
