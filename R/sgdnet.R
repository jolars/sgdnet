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
#'     \frac{\mathrm{change~in~weights}}{\mathrm{weights}} < \mathrm{thresh}.
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

  y <- as.matrix(y)

  debug <- getOption("sgdnet.debug")

  stopifnot(identical(NROW(y), NROW(x)),
            !any(is.na(y)),
            !any(is.na(x)),
            alpha >= 0 && alpha <= 1,
            thresh > 0,
            lambda >= 0,
            is.logical(intercept),
            is.logical(standardize),
            is.logical(debug))

  # Setup reponse type options and assert that input is correct
  switch(
    match.arg(family),
    gaussian = {
      stopifnot(is.numeric(y),
                identical(NCOL(y), 1L))
    }
  )

  # Fit the model by calling the Rcpp routine.
  res <- FitModel(x,
                  y,
                  family,
                  intercept,
                  is_sparse,
                  alpha_sklearn,
                  beta_sklearn,
                  standardize,
                  maxit,
                  thresh,
                  debug)

  variable_names <- colnames(x)

  # Organize return values
  a0 <- drop(res$a0)
  names(a0) <- paste0("s", seq_along(a0) - 1L)
  beta <- Matrix::Matrix(res$beta)
  dimnames(beta) <- list(colnames(x), names(a0))

  out <- structure(list(a0 = a0,
                        beta = beta,
                        npasses = res$npasses),
                   class = "sgdnet")
  if (debug)
    attr(out, "debug_info") <- list(loss = res$losses)
  out
}
