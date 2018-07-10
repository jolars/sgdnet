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

#' Retrieve Response Family from sgdnet Object
#'
#' @param x an object of class `'sgdnet'`
#'
#' @return The family that was used in fitting `x`.
#'
#' @keywords internal
extract_family <- function(x) {
  supported_families <- c("gaussian", "binomial")
  family_index <- inherits(x, paste0("sgdnet_", supported_families), TRUE) > 0
  supported_families[family_index]
}

#' Check if x is FALSE
#'
#' @param x Argument to be tested.
#'
#' @return A bool.
#'
#' @keywords internal
is_false <- function(x) {
  identical(x, FALSE)
}
