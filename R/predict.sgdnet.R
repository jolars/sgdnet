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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.]

# These functions have been adapted from the glmnet R package maintained
# by Trevor Hastie, which is licensed under the GPL-2:
#
# glmnet: Lasso and Elastic-Net Regularized Generalized Linear Models
# Copyright (C) 2018 Jerome Friedman, Trevor Hastie, Rob Tibshirani, Noah Simon
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

#' Return Non-zero Coefficients
#'
#' @param beta ceofficients
#' @param bystep `FALSE` selects variables that were ever nonzero.
#'   `TRUE` selects which variables are nonzero at each step along the
#'   path.
#'
#' @author Jerome Friedman, Trevor Hastie, Rob Tibshirani, and Noah Simon
#'
#' @return Nonzero coefficients.
#' @keywords internal
nonzero_coefs <- function(beta, bystep = FALSE) {

  nr <- nrow(beta)

  if (nr == 1) {
    #degenerate case
    if (bystep)
      apply(beta, 2, function(x) if (abs(x) > 0) 1 else NULL)
    else {
      if (any(abs(beta) > 0))
        1
      else
        NULL
    }
  } else {
    beta <- abs(beta) > 0 # this is sparse
    which <- seq(nr)
    ones <- rep(1, ncol(beta))
    nz <- as.vector((beta %*% ones) > 0)
    which <- which[nz]

    if (bystep) {

      if (length(which) > 0) {

        beta <- as.matrix(beta[which, , drop = FALSE])
        nzel <- function(x, which) if (any(x)) which[x] else NULL
        which <- apply(beta, 2, nzel, which)

        if (!is.list(which))
          which <- data.frame(which)

        which

      } else {

        dn <- dimnames(beta)[[2]]
        which <- vector("list", length(dn))
        names(which) <- dn
        which

      }
    } else which
  }
}

#' Interpolate Lambda Values
#'
#' Interpolate
#'
#' @param lambda lambda penalty
#' @param s the lambda penalty wanted by the user
#'
#' @return Interpolated values of lambda
#' @author Jerome Friedman, Trevor Hastie, Rob Tibshirani, and Noah Simon
#'
#' @keywords internal
lambda_interpolate <- function(lambda, s) {

  if (length(lambda) == 1) {

    nums <- length(s)
    left <- rep(1, nums)
    right <- left
    sfrac <- rep(1, nums)

  } else{

    s[s > max(lambda)] = max(lambda)
    s[s < min(lambda)] = min(lambda)
    k <- length(lambda)
    sfrac <- (lambda[1] - s)/(lambda[1] - lambda[k])
    lambda <- (lambda[1] - lambda)/(lambda[1] - lambda[k])
    coord <- stats::approx(lambda, seq(lambda), sfrac)$y
    left <- floor(coord)
    right <- ceiling(coord)
    sfrac <- (sfrac - lambda[right])/(lambda[left] - lambda[right])
    sfrac[left == right] <- 1
    sfrac[abs(lambda[left] - lambda[right]) < .Machine$double.eps] <- 1
  }

  list(left = left, right = right, frac = sfrac)
}

#' Predictions for sgdnet Models
#'
#' @param object an object of class `'sgdnet'`.
#' @param newx new data to predict on. Must be provided if `type` is
#'   `"link"`.
#' @param s the lambda penalty value on which to base the predictions.
#' @param type type of prediction to return, one of
#' \describe{
#'   \item{`link`}{ linear predictors,}
#'   \item{`response`}{responses,}
#'   \item{`coefficients`}{coefficients (weights); equivalent to calling
#'     [coef()], and}
#'   \item{`nonzero`}{nonzero coefficients at each step of the regularization
#'     path.}
#'  }
#' @param exact if the given value of `s` is not in the model and
#'   `exact = TRUE`, the model will be refit using `s`. If `FALSE`, predictions
#'   will be made using a linearly interpolated coefficient matrix.
#' @param ... arguments to be passed on to [stats::update()] to refit
#'   the model via [sgdnet()] if `s` is missing
#'   from the model and an exact fit is required by `exact`.
#'
#' @return Predictions for `object` given data in `newx`.
#' @export
#'
#' @author Jerome Friedman, Trevor Hastie, Rob Tibshirani, Noah Simon
#'   (original), Johan Larsson (modifications)
#'
#' @seealso [sgdnet()], [coef.sgdnet()]
#'
#' @examples
#' # Gaussian
#' id <- sample.int(nrow(iris))
#' train_ind <- id[1:100]
#' test_ind <- id[101:150]
#' gaussian_fit <- sgdnet(iris[train_ind, 2:4], iris[train_ind, 1])
#' pred <- predict(gaussian_fit,
#'                 newx = iris[test_ind, 2:4],
#'                 s = 4.23,
#'                 type = "response",
#'                 exact = TRUE)
#'
predict.sgdnet <- function(object,
                           newx,
                           s = NULL,
                           type = c("link",
                                    "response",
                                    "coefficients",
                                    "nonzero"),
                           exact = FALSE,
                           ...) {

  stopifnot(is.logical(exact))

  type <- match.arg(type)

  if (missing(newx) && type %in% c("link"))
    stop(paste("new data must be provided for type =", type))

  if (isTRUE(exact) && !is.null(s)) {
    lambda <- object$lambda
    which <- match(s, lambda, FALSE)

    if (!all(which > 0)) {
      lambda <- unique(rev(sort(c(s, lambda))))
      object <- stats::update(object, lambda = lambda)
    }
  }

  a0 <- t(as.matrix(object$a0))
  rownames(a0) <- "(Intercept)"
  beta <- methods::rbind2(a0, object$beta)

  if (!is.null(s)) {
    stopifnot(s >= 0)

    vnames <- dimnames(beta)[[1]]
    dimnames(beta) <- list(NULL, NULL)
    lambda <- object$lambda
    lamlist <- lambda_interpolate(lambda, s)

    beta <- beta[, lamlist$left, drop = FALSE] %*%
      Matrix::Diagonal(x = lamlist$frac) +
      beta[, lamlist$right, drop = FALSE] %*%
      Matrix::Diagonal(x = 1 - lamlist$frac)

    dimnames(beta) = list(vnames, paste(seq(along = s)))
  }

  switch(
    type,
    link = {
      if (inherits(newx, "sparseMatrix"))
        newx <- methods::as(newx, "dgCMatrix")
      else
        newx <- as.matrix(newx)

      if (inherits(object, "gaussian"))
        as.matrix(cbind(1, newx) %*% beta)
    },
    response = {
      if (inherits(newx, "sparseMatrix"))
        newx <- methods::as(newx, "dgCMatrix")
      else
        newx <- as.matrix(newx)

      if (inherits(object, "gaussian"))
        as.matrix(cbind(1, newx) %*% beta)
    },
    coefficients = beta,
    nonzero = nonzero_coefs(beta[-1, , drop = FALSE], bystep = TRUE)
  )
}

#' Extract Model Coefficients for sgdnet Model
#'
#' This is simply a wrapper for [predict.sgdnet()] with `type = "coefficients"`.
#'
#' @param object a model of class `'sgdnet'`, typically from a call to
#'   [sgdnet()].
#' @param ... passed on to [predict.sgdnet()]
#'
#' @return A sparse matrix with intercept in the first row and betas
#'   in the rest.
#'
#' @seealso [predict.sgdnet()], [sgdnet()]
#'
#' @export
#'
#' @examples
#' fit <- sgdnet(matrix(rnorm(100), 50, 2), rnorm(50))
#' coef(fit)
coef.sgdnet <- function(object, ...) {
  stats::predict(object, type = "coefficients", ...)
}
