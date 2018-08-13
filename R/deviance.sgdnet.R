#' Deviance for sgdnet Object
#'
#' Return the deviance for an object of class `"sgdnet"`, typically
#' from a fit with [sgdnet()].
#'
#' This functions returns the deviance of the model along the regularization
#' path. It is computed from the slots `dev.ratio` and `nulldev` from the
#' `"sgdnet"` object using the formula
#'
#' \deqn{
#'   (1 - \mathtt{dev.ratio}) \times \mathtt{nulldev}
#' }{
#'   (1 - dev.ratio)*nulldev
#' }
#'
#' where `nulldev` is the deviance of the intercept-only model.
#'
#' @param object an object of class `'sgdnet'`
#' @param ... ignored
#'
#' @return The deviance of `object` at each value along the regularization path.
#'   For `family = "gaussian"` in [sgdnet()], this is the residual sum of
#'   squares.
#'
#' @export
#'
#' @seealso [stats::deviance()], [sgdnet()]
#'
#' @examples
#' fit <- sgdnet(wine$x, wine$y, family = "multinomial")
#' deviance(fit)
deviance.sgdnet <- function(object, ...) {
  (1 - object$dev.ratio)*object$nulldev
}

#' @export
#' @rdname deviance
deviance.cv_sgdnet <- function(object, ...) {
  sats::deviance(object$fit, ...)
}
