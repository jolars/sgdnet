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

#' Abalone
#'
#' This data set contains observations of abalones, the common
#' name for any of a group of sea snails. The goal is to predict the
#' age of an individual abalone given physical measurements such as
#' sex, weight, and height.
#'
#' @format A list with two items representing 20,640 observations from
#'   9 variables
#' \describe{
#'   \item{sex}{sex of abalone, 1 for female}
#'   \item{infant}{indicates that the person is an infant}
#'   \item{length}{longest shell measurement in mm}
#'   \item{diameter}{perpendicular to length in mm}
#'   \item{height}{height in mm including meat in shell}
#'   \item{weight_whole}{weight of entire abalone}
#'   \item{weight_shucked}{weight of meat}
#'   \item{weight_viscera}{weight of viscera}
#'   \item{weight_shell}{weight of shell}
#'   \item{rings}{rings. +1.5 gives the age in years}
#' }
#' @source Pace, R. Kelley and Ronald Barry, Sparse Spatial Autoregressions,
#'   Statistics and Probability Letters, 33 (1997) 291-297.
"abalone"

#' Heart Disease
#'
#' Diagnostic attributes of patients classified as having heart disease or not.
#'
#' @section Preprocessing:
#' The original dataset contained 13 variables. The nominal of these were
#' dummycoded, removing the first category. No precise information regarding
#' variables `chest_pain`, `thal` and `ecg` could be found, which explains
#' their obscure definitions here.
#'
#' @format 270 observations from 17 variables represented as a list consisting
#'  of a binary factor response vector `y`,
#' with levels 'absence' and 'presence' indicating the absence or presence of
#' heart disease and `x`: a sparse feature matrix of class 'dgCMatrix' with the
#' following variables:
#' \describe{
#'   \item{age}{age}
#'   \item{bp}{diastolic blood pressure}
#'   \item{chol}{serum cholesterol in mg/dl}
#'   \item{hr}{maximum heart rate achieved}
#'   \item{old_peak}{ST depression induced by exercise relative to rest}
#'   \item{vessels}{the number of major blood vessels (0 to 3) that were
#'                  colored by fluoroscopy}
#'   \item{sex}{sex of the participant: 0 for male, 1 for female}
#'   \item{angina}{a dummy variable indicating whether the person suffered
#'                 angina-pectoris during exercise}
#'   \item{glucose_high}{indicates a fasting blood sugar over 120 mg/dl}
#'   \item{cp_typical}{typical angina}
#'   \item{cp_atypical}{atypical angina}
#'   \item{cp_nonanginal}{non-anginal pain}
#'   \item{ecg_abnormal}{indicates a ST-T wave abnormality
#'                       (T wave inversions and/or ST elevation or depression of
#'                       > 0.05 mV)}
#'   \item{ecg_estes}{probable or definite left ventricular hypertrophy by
#'                    Estes' criteria}
#'   \item{slope_flat}{a flat ST curve during peak exercise}
#'   \item{slope_downsloping}{a downwards-sloping ST curve during peak exercise}
#'   \item{thal_reversible}{reversible defect}
#'   \item{thal_fixed}{fixed defect}
#' }
#' @source Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository
#'   <http://archive.ics.uci.edu/ml>. Irvine, CA: University of California,
#'   School of Information and Computer Science.
#' @source <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#heart>
"heart"

#' Wine Cultivars
#'
#' A data set of results from chemical analysis of wines grown in Italy
#' from three different cultivars.
#'
#' @format 178 observations from 13 variables represented as a list consisting
#'  of a binary factor response vector `y`
#' with three levels: *A*, *B*, and *C* representing different
#' cultivars of wine as well as `x`: a sparse feature matrix of class
#' 'dgCMatrix' with the following variables:
#' \describe{
#'   \item{alcohol}{alcoholic content}
#'   \item{malic}{malic acid}
#'   \item{ash}{ash}
#'   \item{alcalinity}{alcalinity of ash}
#'   \item{magnesium}{magnemium}
#'   \item{phenols}{total phenols}
#'   \item{flavanoids}{flavanoids}
#'   \item{nonflavanoids}{nonflavanoid phenols}
#'   \item{proanthocyanins}{proanthocyanins}
#'   \item{color}{color intensity}
#'   \item{hue}{hue}
#'   \item{dilution}{OD280/OD315 of diluted wines}
#'   \item{proline}{proline}
#' }
#'
#' @source Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository
#'   <http://archive.ics.uci.edu/ml>. Irvine, CA: University of California,
#'   School of Information and Computer Science.
#' @source <https://raw.githubusercontent.com/hadley/rminds/master/1-data/wine.csv>
#' @source <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#wine>
"wine"

#' Benchmark data
#'
#' @source <https://github.com/jolars/sgdnet/data-raw/
#'
#' @format A `list` with a `data.frame` for each model family (Gaussian, binomial,
#'   multinomial), featuring 5 variables:
#' \describe{
#'   \item{dataset}{dataset used}
#'   \item{penalty}{type of penalty (ridge or lasso)}
#'   \item{package}{R package}
#'   \item{time}{run time in seconds}
#'   \item{loss}{objective loss}
#' }
"benchmarks"
