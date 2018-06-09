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

#' @useDynLib sgdnet, .registration = TRUE
#' @importFrom Rcpp sourceCpp
NULL

#' Penalized Generalized Linear Models with Stochastic Gradient Descent
#'
#' @section Package options:
#'
#' Parts of the [sgdnet] API can be interfaced with via
#' \code{\link{options}} as follows:
#'
#' \itemize{
#'   \item \code{sgdnet.debug}: set to `TRUE` to enable debugging features,
#'                              which will be accessible via a `'diagnostics'`
#'                              attribute to the `'sgdnet'` object
#'                              that is returned by [sgdnet()].
#' }
#' @name sgdnet-package
#' @docType package
NULL
