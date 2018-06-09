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

#' Fit a Generalized Linear Model with Elastic Net Regularization
#'
#' @section Regularization Path:
#' The default regularization path is a sequence of `nlambda`
#' log-spaced elements
#' from \eqn{\lambda_{\mathrm{max}}}{lambda_max} to
#' \eqn{\lambda_{\mathrm{max}} \times \mathtt{lambda.min.ratio}}{
#'      lambda_max*lambda.min.ratio},
#' For the gaussian family, for instance,
#' \eqn{\lambda_{\mathrm{max}}}{lambda_max} is
#' the largest absolute inner product of the feature vectors and the response
#' vector,
#' \deqn{\max_i \frac{1}{n}|\langle\mathbf{x}_i, y\rangle|.}{
#'       max |<x, y>|/n.}
#'
#' @param x input matrix
#' @param y response variable
#' @param family reponse type
#' @param alpha elastic net mixing parameter
#' @param nlambda number of penalties in the regualrization path
#' @param lambda.min.ratio the ratio between `lambda_max` (the smallest
#'   penalty at which the solution is completely sparse) and the smallest
#'   lambda value on the path. See section **Regularization Path** for details.
#' @param lambda regularization strength
#' @param intercept whether to fit an intercept or not
#' @param maxit maximum number of effective passes (epochs)
#' @param standardize whether to standardize `x` or not -- ignored when
#'   `intercept == FALSE`.
#' @param thresh tolerance level for termination of the algorithm. The
#'   algorithm terminates when
#'   \deqn{
#'     \frac{|\beta^{(t)} - \beta^{(t-1)}|{\infty}}{|\beta^{(t)}|{\infty}} < \mathrm{thresh}
#'   }{
#'     max(change in weights)/max(weights) < thresh.
#'   }
#' @param ... ignored
#'
#' @return An object of class `'sgdnet'`.
#'
#' @seealso [predict.sgdnet()], [plot.sgdnet()], [coef.sgdnet()],
#'   [sgdnet-package()]
#'
#' @export
#'
#' @examples
#' # Gaussian regression with sparse features
#' fit <- sgdnet(permeability$x, permeability$y, alpha = 0)
sgdnet <- function(x, ...) UseMethod("sgdnet")

#' @export
#' @rdname sgdnet
sgdnet.default <- function(x,
                           y,
                           family = c("gaussian"),
                           alpha = 1,
                           nlambda = 100,
                           lambda.min.ratio =
                             if (NROW(x) < NCOL(x)) 0.01 else 0.0001,
                           lambda = NULL,
                           maxit = 1000,
                           standardize = TRUE,
                           intercept = TRUE,
                           thresh = 0.001,
                           ...) {

  # Collect the call so we can use it in update() later on
  ocall <- match.call(call = sys.call(1))

  n_samples <- NROW(x)
  n_features <- NCOL(x)

  # Convert sparse x to dgCMatrix class from package Matrix.
  if (is_sparse <- inherits(x, "sparseMatrix")) {
    x <- methods::as(x, "dgCMatrix")
  } else {
    x <- as.matrix(x)
  }

  # Collect response and variable names (if they are given) and otherwise
  # make new.
  response_names <- colnames(y)
  variable_names <- colnames(x)

  if (is.null(variable_names))
    variable_names <- paste0("V", seq_len(NCOL(x)))
  if (is.null(variable_names))
    response_names <- paste0("y", seq_len(NCOL(y)))

  y <- as.matrix(y)

  # Collect sgdnet-specific options for debugging and more
  debug <- getOption("sgdnet.debug")

  if (is.null(lambda))
    lambda <- double(0L)

  stopifnot(identical(NROW(y), NROW(x)),
            !any(is.na(y)),
            !any(is.na(x)),
            alpha >= 0 && alpha <= 1,
            length(alpha) == 1L,
            thresh > 0,
            all(lambda >= 0),
            is.logical(intercept),
            is.logical(standardize),
            is.logical(debug))

  # Setup reponse type options and assert appropriate input
  family <- match.arg(family)

  switch(family,
         gaussian = {
           stopifnot(is.numeric(y),
                     identical(NCOL(y), 1L))
           }
         )

  control <- list(family = family,
                  intercept = intercept,
                  is_sparse = is_sparse,
                  lambda = lambda,
                  elasticnet_mix = alpha,
                  n_lambda = nlambda,
                  lambda_min_ratio = lambda.min.ratio,
                  normalize = standardize,
                  max_iter = maxit,
                  tol = thresh,
                  debug = debug)

  # Fit the model by calling the Rcpp routine.
  res <- SgdnetCpp(x, y, control)

  # Setup return values
  a0 <- drop(t(as.matrix(res$a0)))
  beta <- res$beta
  lambda <- drop(res$lambda)
  n_penalties <- length(lambda)

  path_names <- paste0("s", seq_along(lambda) - 1L)

  beta <- lapply(seq(dim(beta)[2L]), function(x) {
      Matrix::Matrix(as.vector(beta[, x, ]),
                     nrow = n_features,
                     ncol = n_penalties,
                     dimnames = list(variable_names,
                                     path_names),
                     sparse = TRUE)
    }
  )

  if (family %in% c("gaussian", "binomial", "poisson", "cox")) {
    # NOTE(jolars): I would rather not to this, i.e. have different outputs
    # depending on family, but this makes it equivalent to glmnet output
    beta <- beta[[1L]]
    names(a0) <- path_names
  } else {
    a0 <- t(a0)
    rownames(a0) <- response_names
    colnames(a0) <- path_names
  }

  out <- structure(list(a0 = a0,
                        beta = beta,
                        npasses = res$npasses,
                        lambda = lambda,
                        alpha = alpha,
                        call = ocall),
                   class = c("sgdnet", family))
  if (debug)
    attr(out, "diagnostics") <- list(loss = res$losses)
  out
}
