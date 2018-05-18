#' Extract Model Coefficients for sgdnet Model
#'
#' @param object a model of class `'sgdnet'`, typically from a call to
#'   [sgdnet()].
#' @param ... ignored
#'
#' @return A sparse matrix with intercept in the first row and betas
#'   in the rest.
#' @export
#'
#' @examples
#' fit <- sgdnet(matrix(rnorm(100), 50, 2), rnorm(50))
#' coef(fit)
coef.sgdnet <- function(object, ...) {
  out <- rbind(object$a0, object$beta)
  varnames <- rownames(object$beta)
  if (is.null(varnames))
    varnames <- seq_len(NROW(object$beta))

  dimnames(out) <- list(c("(Intercept)", varnames),
                        colnames(object$beta))
  out
}
