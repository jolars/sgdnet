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

#' Check if x is FALSE
#'
#' @param x Argument to be tested.
#'
#' @return A bool.
#'
#' @keywords internal
#' @noRd
is_false <- function(x) {
  identical(x, FALSE)
}

#' Standard Deviation by Columns
#'
#' @param x a matrix or data.frame
#' @param na.rm whether to remove NA values
#'
#' @return A vector of standard deviations, one per column
#'
#' @noRd
#' @keywords internal
col_sd <- function(x, na.rm = TRUE) {
  if (na.rm) {
    n <- colSums(!is.na(x))
  } else {
    n <- nrow(x)
  }
  col_var <- colMeans(x^2, na.rm = na.rm) - (colMeans(x, na.rm = na.rm))^2
  sqrt(col_var*n/(n - 1))
}
