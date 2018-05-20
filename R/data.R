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
