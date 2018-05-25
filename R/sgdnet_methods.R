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
  beta <- object$beta
  a0 <- object$a0

  if (is.list(beta)) {
    for (i in seq_along(beta)) {
      beta[[i]] <- rbind(a0[i, ], beta[[i]])
      rownames(beta)[[i]][1L] <- "(Intercept)"
    }
  } else {
    beta <- rbind(a0, beta)
    rownames(beta)[1L] <- "(Intercept)"
  }

  beta
}
