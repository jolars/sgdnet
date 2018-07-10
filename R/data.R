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

#' Permeability
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

#' Mushrooms
#'
#' Mushroom records from The Audubon Society Field Guide to North
#' American Mushrooms (1981). G. H. Lincoff (Pres.), New York: Alfred A. Knopf.
#'
#' The data set was retrieved from the UCI database and came with the
#' following description:
#'
#' "This data set includes descriptions of hypothetical samples corresponding to
#' 23 species of gilled mushrooms in the Agaricus and Lepiota Family.
#' Each species is identified as definitely edible, definitely
#' poisonous, or of unknown edibility and not recommended. This latter class was
#' combined with the poisonous one. The Guide clearly states that there is no
#' simple rule for determining the edibility of a mushroom; no rule like
#' "leaflets three, let it be" for Poisonous Oak and Ivy."
#'
#' @section Processing:
#' The original dataset contained 22 variables (features). Out of these,
#' "stalk-root" was dropped since it contained missing values. All of the
#' remaining variables were dummy-coded.
#'
#' @format A list with two items representing 8,124 observations from
#'   112 variables.
#' \describe{
#'   \item{x}{a sparse feature matrix dummy-coded attributes for each
#'            mushroom}
#'   \item{y}{a factor variable represing the outcome, edibility, either
#'            "edible" or "poisonous"}
#' }
#' @source The Audubon Society Field Guide to North
#'   American Mushrooms (1981). G. H. Lincoff (Pres.), New York: Alfred A.
#'   Knopf.
#' @source <https://archive.ics.uci.edu/ml/datasets/mushroom>
#' @source Dua, D. and Karra Taniskidou, E. (2017).
#'   UCI Machine Learning Repository <http://archive.ics.uci.edu/ml>. Irvine,
#'   CA: University of California, School of Information and Computer Science.
"mushrooms"

#' Benchmark data for binomial response family
#'
#' @source <https://github.com/jolars/sgdnet/data-raw/
#'
#' @format A `data.frame` with 5 variables and 2000 observations.
#' \describe{
#'   \item{dataset}{dataset used}
#'   \item{package}{R package}
#'   \item{time}{run time in seconds}
#'   \item{loss}{objective loss}
#'   \item{penalty}{type of penalty, "ridge" or "lasso"}
#' }
"benchmarks_binomial"
