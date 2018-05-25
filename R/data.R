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

#' Permeability data
#'
#' A pharmaceutical dataset of permeability values for 165 compounds
#' for which molecular fingerprints were collected and represented
#' as binary indices.
#'
#' @format A list with two items representing 165 observations from
#'   1107 variables.
#' \describe{
#'   \item{x}{a sparse feature matrix with binary indicators for molecular
#'            fingerprints labeled X1 to X1107}
#'   \item{y}{a numeric vector of permeability values for 165 compounds}
#' }
#' @source [AppliedPredictiveModeling::permeability], which features a
#'   thorough description on the type of data.
"permeability"
