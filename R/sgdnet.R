#' Fit a Generalized Linear Model with Elasticnet Regularization
#'
#' @param x input matrix
#' @param y response variable
#' @param family reponse type
#' @param alpha elasticnet mixing parameter
#' @param lambda regularization strength
#' @param maxit maximum number of effective passes (epochs)
#' @param standardize whether to standardize `x` or not -- ignored when
#'   `intercept == TRUE`.
#' @param thresh tolerance level for termination of the algorithm
#' @param ... ignored
#'
#' @return An object of class `'sgdnet'`.
#' @export
#'
#' @examples
#' x <- rnorm(30, 10, 3)
#' y <- rnorm(10)
#' sgdnet(x, y)
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
    x <- as(x, "CsparseMatrix")
    x <- as(x, "dgCMatrix")
  }

  # Fit the model by calling the Rcpp routine.
  res <- FitModel(x,
                  as.matrix(y),
                  match.arg(family),
                  intercept,
                  is_sparse,
                  alpha_sklearn,
                  beta_sklearn,
                  standardize,
                  maxit,
                  thresh)

  variable_names <- colnames(x)

  # Organize return values
  a0 <- drop(res$a0)
  names(a0) <- paste0("s", seq_along(a0) - 1L)
  beta <- Matrix::Matrix(res$beta)
  dimnames(beta) <- list(colnames(x), names(a0))

  structure(list(a0 = a0,
                 beta = beta,
                 npasses = res$npasses),
            class = "sgdnet")
}
