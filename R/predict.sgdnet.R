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
#' @noRd
nonzero_coefs <- function(beta, bystep = FALSE) {
  if (is.list(beta)) {
    lapply(beta, nonzero_coefs, bystep = bystep)
  } else {
    nr <- nrow(beta)

    if (nr == 1) {
      # degenerate case
      if (bystep)
        apply(beta, 2, function(x) if (abs(x) > 0) 1 else NULL)
      else {
        if (any(abs(beta) > 0)) 1 else NULL
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
}

#' Softmax-based prediction
#'
#' @param x linear predictors
#'
#' @return Class predictions.
#' @keywords internal
softmax <- function(x) {
  d <- dim(x)
  nas <- apply(is.na(x), 1, any)
  if (any(nas)) {
    pclass <- rep(NA, d[1])
    if (sum(nas) < d[1]) {
      pclass2 <- softmax(x[!nas, ])
      pclass[!nas] <- pclass2
      if (is.factor(pclass2))
        pclass <- factor(pclass, levels = seq(d[2]), labels = levels(pclass2))
    }
  } else {
    maxdist <- x[, 1]
    pclass <- rep(1, d[1])
    for (i in seq(2, d[2])) {
      l <- x[, i] > maxdist
      pclass[l] <- i
      maxdist[l] <- x[l, i]
    }
    dd <- dimnames(x)[[2]]
    pclass <- if (is.null(dd) || !length(dd))
      pclass
    else
      factor(pclass, levels = seq(d[2]), labels = dd)
  }
  pclass
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
#' @noRd
lambda_interpolate <- function(lambda, s) {

  if (length(lambda) == 1) {

    nums <- length(s)
    left <- rep(1, nums)
    right <- left
    sfrac <- rep(1, nums)

  } else {

    s[s > max(lambda)] <- max(lambda)
    s[s < min(lambda)] <- min(lambda)
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

#' Refit sgdnet object
#'
#' @param object an object of class `sgdnet`
#' @param s a new lambda penalty (or several)
#' @param ... new arguments to pass on to [sgdnet()]
#'
#' @return A new fit from [sgdnet()].
#' @keywords internal
#' @noRd
refit <- function(object, s, ...) {
  if (!all(s %in% object$lambda)) {
    lambda <- unique(rev(sort(c(s, object$lambda))))
    object <- stats::update(object, lambda = object$lambda, ...)
  }
}


#' Bind intercept with coefficients
#'
#' @param beta coefficients
#' @param a0 intercept
#'
#' @return A matrix (or listof matrices).
#' @keywords internal
#' @noRd
bind_intercept <- function(beta, a0) {
  if (is.list(beta)) {
    for (i in seq_along(beta))
      beta[[i]] <- bind_intercept(beta[[i]], a0[i, , drop = FALSE])
    beta
  } else {
    out <- methods::rbind2(a0, beta)
    rownames(out)[1] <- "(Intercept)"
    out
  }
}

#' Interpolate coefficients
#'
#' @param beta coeffieicnts
#' @param s lambda penalty
#' @param lamlist a list of lambda interpolation parameters, returned from
#'   [lambda_interpolate()].
#'
#' @return A matrix (or list of matrices) with new coefficients based
#'   on linearly interpolating from new and old lambda values.
#' @keywords internal
#' @noRd
interpolate_coefficients <- function(beta, s, lamlist) {
  if (is.list(beta)) {
    lapply(beta, interpolate_coefficients, s = s, lamlist = lamlist)
  } else {
    vnames <- dimnames(beta)[[1]]
    dimnames(beta) <- list(NULL, NULL)
    beta <- beta[, lamlist$left, drop = FALSE] %*%
      Matrix::Diagonal(x = lamlist$frac) +
      beta[, lamlist$right, drop = FALSE] %*%
      Matrix::Diagonal(x = 1 - lamlist$frac)
    dimnames(beta) <- list(vnames, paste(seq(along = s)))
    beta
  }
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
#'     [coef()]}
#'   \item{`nonzero`}{nonzero coefficients at each step of the regularization
#'     path, and}
#'   \item{`class`}{class predictions for each new data point in `newx` at
#'     each step of the regularization path -- only useful for 'binomial' and
#'     'multinomial' families.}
#'  }
#' @param exact if the given value of `s` is not in the model and
#'   `exact = TRUE`, the model will be refit using `s`. If `FALSE`, predictions
#'   will be made using a linearly interpolated coefficient matrix.
#' @param newoffset if an offset was used in the call to [sgdnet()],
#'   a new offset can be provided here for making predictions (but not for
#'   `type = 'coefficients'/'nonzero'`)
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
#'
#' # Split into training and test sets
#' n <- length(houses$y)
#' train_ind <- sample(n, size = floor(0.8 * n))
#'
#' # Fit the model using the training set
#' fit_gaussian <- sgdnet(houses$x[train_ind, ], houses$y[train_ind])
#'
#' # Predict using the test set
#' pred_gaussian <- predict(fit_gaussian, newx = houses$x[-train_ind, ])
#'
#' # Mean absolute prediction error along regularization path
#' mae <- 1/(n - length(train_ind)) *
#'          colSums(abs(houses$y[-train_ind] - pred_gaussian))
#'
#' # Binomial
#' n <- length(mushrooms$y)
#' train_ind <- sample(n, size = floor(0.8 * n))
#'
#' fit_binomial <- sgdnet(mushrooms$x[train_ind, ],
#'                        mushrooms$y[train_ind],
#'                        family = "binomial")
#'
#' # Predict classes at custom lambda value (s) using linear interpolation
#' predict(fit_binomial, mushrooms$x[-train_ind, ], type = "class", s = 1/n)
#'
#' # Multinomial
#' fit_multinomial <- sgdnet(pendigits$train$x,
#'                           pendigits$train$y,
#'                           family = "multinomial",
#'                           alpha = 0.25)
#' predict(fit_multinomial,
#'         pendigits$test$x,
#'         s = 1/nrow(pendigits$test$x),
#'         exact = TRUE,
#'         type = "class")
#'
predict.sgdnet <- function(object,
                           newx = NULL,
                           s = NULL,
                           type,
                           exact = FALSE,
                           newoffset = NULL,
                           ...) {

  if (isTRUE(exact) && !is.null(s))
    object <- refit(object, s, ...)

  beta <- bind_intercept(object$beta, object$a0)

  if (!is.null(s)) {
    if (s < 0)
      stop("s (lambda penalty) cannot be negative")

    lamlist <- lambda_interpolate(object$lambda, s)
    beta <- interpolate_coefficients(beta, s, lamlist)
  }

  if (is.null(newx)) {
    if (type %in% c("link", "response", "class"))
      stop("you need to supply a value for 'newx' for type = '", type, "'")
  } else {
    if (inherits(newx, "sparseMatrix"))
      newx <- methods::as(newx, "dgCMatrix")

    fit <- as.matrix(methods::cbind2(1, newx) %*% beta)
  }

  if (isTRUE(object$offset)) {
    if (is.null(newoffset))
      stop("need 'newoffset' since offset was used in fit")
    if (is.matrix(newoffset) &&
        inherits(object, "sgdnet_binomial") &&
        dim(newoffset)[[2]] == 2)
      newoffset <- newoffset[, 2]
    fit <- fit + array(newoffset, dim = dim(fit))
  }

  switch(
    type,
    link = fit,
    response = fit,
    coefficients = beta,
    nonzero = nonzero_coefs(beta[-1, , drop = FALSE], bystep = TRUE),
    fit
  )
}

#' @inherit predict.sgdnet
#'
#' @export
#' @rdname predict.sgdnet
predict.sgdnet_gaussian <- function(object,
                                    newx = NULL,
                                    s = NULL,
                                    type = c("link",
                                             "response",
                                             "coefficients",
                                             "nonzero"),
                                    exact = FALSE,
                                    newoffset = NULL,
                                    ...) {
  type <- match.arg(type)
  NextMethod("predict", type = type)
}

#' @inherit predict.sgdnet
#'
#' @export
#' @rdname predict.sgdnet
predict.sgdnet_binomial <- function(object,
                                    newx = NULL,
                                    s = NULL,
                                    type = c("link",
                                             "response",
                                             "coefficients",
                                             "nonzero",
                                             "class"),
                                    exact = FALSE,
                                    newoffset = NULL,
                                    ...) {
  type <- match.arg(type)
  fit <- NextMethod("predict", type = type)
  switch(
    type,
    response = 1 / (1 + exp(-fit)),
    class = {
      cnum <- ifelse(fit > 0, 2, 1)
      clet <- object$classnames[cnum]
      if (is.matrix(cnum))
        clet <- array(clet, dim(cnum), dimnames(cnum))
      clet
    },
    fit
  )
}

#' @inherit predict.sgdnet
#' @export
#' @rdname predict.sgdnet
predict.sgdnet_multinomial <- function(object,
                                       newx = NULL,
                                       s = NULL,
                                       type = c("link",
                                                "response",
                                                "coefficients",
                                                "nonzero",
                                                "class"),
                                       exact = FALSE,
                                       newoffset = NULL,
                                       ...) {
  type <- match.arg(type)

  if (isTRUE(exact) && !is.null(s))
    object <- refit(object, s, ...)

  a0 <- object$a0
  beta <- object$beta
  klam <- dim(a0)
  nclass <- klam[[1]]

  nbeta <- bind_intercept(beta, a0)

  if (is.null(newx)) {
    if (type %in% c("link", "response", "class"))
      stop("you need to supply a value for 'newx' for type = '", type, "'")
  } else {
    if (inherits(newx, "sparseMatrix"))
      newx <- methods::as(newx, "dgCMatrix")
    else
      newx <- as.matrix(newx)
  }

  if (!is.null(s)) {
    if (s < 0)
      stop("s (lambda penalty) cannot be negative")
    nlambda <- length(s)
    lamlist <- lambda_interpolate(object$lambda, s)
    nbeta <- interpolate_coefficients(nbeta, s, lamlist)
  } else {
    nlambda <- length(object$lambda)
  }

  if (type %in% c("link", "response", "class")) {
    dd <- dim(newx)
    if (inherits(newx, "sparseMatrix"))
      newx <- methods::as(newx, "dgCMatrix")

    npred <- nrow(newx)
    dp <- array(0,
                c(nclass, nlambda, npred),
                dimnames = list(names(nbeta),
                                dimnames(nbeta[[1]])[[2]],
                                dimnames(newx)[[1]]))

    for (i in seq(nclass)) {
      fitk <- methods::cbind2(1, newx) %*% nbeta[[i]]
      dp[i, , ] <- dp[i, , ] + t(as.matrix(fitk))
    }

    if (isTRUE(object$offset)) {
      if (is.null(newoffset))
        stop("no newoffset provided for prediction, yet offset used in fit of sgdnet")

      if (!is.matrix(newoffset) || dim(newoffset)[[2]] != nclass)
        stop("dimension of newoffset should be [", npred, nclass, "]")

      toff <- t(newoffset)
      for (i in seq(nlambda))
        dp[, i, ] <- dp[, i, ] + toff
    }
  }

  switch(
    type,
    coefficients = nbeta,
    nonzero = {
      if (object$grouped)
        nonzero_coefs(object$beta[[1]], bystep = TRUE)
      else
        nonzero_coefs(object$beta, bystep = TRUE)
    },
    response = {
      pp <- exp(dp)
      psum <- apply(pp, c(2, 3), sum)
      aperm(pp / rep(psum, rep(nclass, nlambda * npred)), c(3, 1, 2))
    },
    link = aperm(dp, c(3, 1, 2)),
    class = {
      dp <- aperm(dp, c(3, 1, 2))

      apply(dp, 3, softmax)
    }
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
 soft <- function(x) {
   exp_x <- exp(x)
   exp_x/sum(exp_x)
 }

