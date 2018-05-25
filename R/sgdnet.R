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
#' @param x input matrix
#' @param y response variable
#' @param family reponse type
#' @param alpha elastic net mixing parameter
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
#' @export
#'
#' @examples
#' # Gaussian regression with sparse features
#' fit <- sgdnet(permeability$x, permeability$y, alpha = 0)
sgdnet <- function(x, y, ...) UseMethod("sgdnet")

#' @export
#' @rdname sgdnet
sgdnet.default <- function(x,
                           y,
                           family = c("gaussian"),
                           alpha = 1,
                           lambda = 1/NROW(x),
                           maxit = 1000,
                           standardize = TRUE,
                           intercept = TRUE,
                           thresh = 0.001,
                           ...) {

  n_samples <- NROW(x)

  # The internal optimizer uses a different penalty construction than
  # glmnet. Convert lambda vallues to match alpha and beta from scikit-learn.
  alpha_sklearn <- lambda/2*n_samples*(1 - alpha)
  beta_sklearn <- lambda/2*n_samples*alpha

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

  stopifnot(identical(NROW(y), NROW(x)),
            !any(is.na(y)),
            !any(is.na(x)),
            alpha >= 0 && alpha <= 1,
            length(alpha) == 1L,
            thresh > 0,
            lambda >= 0,
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
                  alpha = alpha_sklearn,
                  beta = beta_sklearn,
                  normalize = standardize,
                  max_iter = maxit,
                  tol = thresh,
                  debug = debug)

  # Fit the model by calling the Rcpp routine.
  res <- SgdnetCpp(x, y, control)

  # Setup return values

  a0 <- t(as.matrix(res$a0))
  beta <- res$beta

  colnames(a0) <- paste0("s", seq_along(lambda) - 1L)
  dimnames(beta) <- list(variable_names, response_names, rownames(a0))

  beta <- lapply(seq(dim(beta)[2L]),
                 function(x) methods::as(as.matrix(beta[ , x, ]), "dgCMatrix"))

  if (family %in% c("gaussian", "binomial", "poisson", "cox")) {
    # NOTE(jolars): I would rather not to this, i.e. have different outputs
    # depending on family, but this is what they do in glmnet.
    beta <- beta[[1L]]
  } else {
    rownames(a0) <- response_names
  }

  out <- structure(list(a0 = a0,
                        beta = beta,
                        npasses = res$npasses,
                        lambda = lambda,
                        alpha = alpha),
                   class = "sgdnet")
  if (debug)
    attr(out, "diagnostics") <- list(loss = res$losses)
  out
}
