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
#'                              which will be accessible via a `'debug_info'`
#'                              attribute to the `'sgdnet'` object
#'                              that is returned by [sgdnet()].
#' }
#' @name sgdnet-package
#' @docType package
NULL
