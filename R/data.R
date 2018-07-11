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

#' S&P Letters Data
#'
#' Spatial data containing housing prices with economic covariates.
#'
#' "We collected information on the variables using all the block groups in
#' California from the 1990 Census. In this sample a block group on average
#' includes 1425.5 individuals living in a geographically compact area.
#' Naturally, the geographical area included varies inversely with the
#' population density. We computed distances among the centroids of each
#' block group as measured in latitude and longitude. We excluded all the block
#' groups reporting zero entries for the independent and dependent variables.
#' The final data contained 20,640 observations on 9 variables. The dependent
#' variable is ln(median house value)."
#'
#' @format A list with two items representing 20,640 observations from
#'   9 variables
#' \describe{
#'   \item{x}{a dense feature matrix with median house value, median income,
#'            housing median age, total rooms, total bedrooms, population,
#'            households, latitude, and longitude}
#'   \item{y}{a numeric vector of permeability values for 165 compounds}
#' }
#' @source Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
#'   Statistics and Probability Letters, 33 (1997) 291-297.
"houses"

#' Mushroom Data Set
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

#' Pen-Based Recognition of Handwritten Digits Data Set
#'
#' A data set of 250 samples of handwritten digits from 44 writers. Thirty of
#' these make up the training set.
#'
#' From description on UCI (with a few formatting changes):
#'
#' "In our study, we use only (x, y) coordinate information. The stylus
#' pressure level values are ignored. First we apply normalization to make our
#' representation invariant to translations and scale distortions. The raw
#' data that we capture from the tablet consist of integer values between 0 and
#' 500 (tablet input box resolution). The new coordinates are such that the
#' coordinate which has the maximum range varies between 0 and 100. Usually x
#' stays in this range, since most characters are taller than they are wide.
#'
#' In order to train and test our classifiers, we need to represent digits as
#' constant length feature vectors. A commonly used technique leading to good
#' results is resampling the \eqn{(x_t, y_t)} points. Temporal
#' resampling (points
#' regularly spaced in time) or spatial resampling (points regularly spaced in
#' arc length) can be used here. Raw point data are already regularly spaced in
#' time but the distance between them is variable. Previous research showed
#' that spatial resampling to obtain a constant number of regularly spaced
#' points on the trajectory yields much better performance, because it provides
#' a better alignment between points. Our resampling algorithm uses simple
#' linear interpolation between pairs of points. The resampled digits are
#' represented as a sequence of T points
#' \eqn{(x_t, y_t)_{t=1}^\mathsf{T}}{(x_t, y_t)_{t=1}^T}, regularly
#' spaced in arc length, as opposed to the input sequence, which is regularly
#' spaced in time.
#'
#' So, the input vector size is 2*T, two times the number of points resampled.
#' We considered spatial resampling to T = 8, 12, 16 points in our experiments
#' and
#' found that T = 8 gave the best trade-off between accuracy and complexity."
#'
#' @source <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html>
#' @source <https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits>
#' @source F. Alimoglu (1996) Combining Multiple Classifiers for Pen-Based
#'   Handwritten Digit Recognition, MSc Thesis, Institute of Graduate Studies
#'   in Science and Engineering, Bogazici University.
#' @source F. Alimoglu, E. Alpaydin, "Methods of Combining Multiple Classifiers
#'   Based on Different Representations for Pen-based Handwriting Recognition,"
#'   Proceedings of the Fifth Turkish Artificial Intelligence and Artificial
#'   Neural Networks Symposium (TAINN 96), June 1996, Istanbul, Turkey.
#'
#' @format A `list` containing two lists, `train` and `test` with
#'   7,494 and 3,498 observations respectively, each containing
#' \describe{
#'   \item{x}{a feature matrix of class `'Matrix::dgCMatrix'` with
#'            16 features}
#'   \item{y}{a factor with 10 levels, one for each digit}
#' }
"pendigits"

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
