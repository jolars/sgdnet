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

#' Extract Model Coefficients for sgdnet Model
#'
#' @param object a model of class `'sgdnet'`, typically from a call to
#'   [sgdnet()].
#' @param ... ignored
#'
#' @return A sparse matrix with intercept in the first row and betas
#'   in the rest.
#' @export
#'
#' @examples
#' fit <- sgdnet(matrix(rnorm(100), 50, 2), rnorm(50))
#' coef(fit)
coef.sgdnet <- function(object, ...) {
  beta <- object$beta
  a0 <- object$a0

  if (is.list(beta)) {
    for (i in seq_along(beta)) {
      beta[[i]] <- rbind(a0[i, ], beta[[i]])
      rownames(beta)[[i]][1L] <- "(Intercept)"
    }
  } else {
    beta <- rbind(a0, beta)
    rownames(beta)[1L] <- "(Intercept)"
  }

  beta
}
