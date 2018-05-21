#' Fit a Generalized Linear Model with Elasticnet Regularization
#'
#' @param x input matrix
#' @param y response variable
#' @param family reponse type
#' @param alpha elasticnet mixing parameter
#' @param lambda regularization strength
#' @param intercept whether to fit an intercept or not
#' @param maxit maximum number of effective passes (epochs)
#' @param standardize whether to standardize `x` or not -- ignored when
#'   `intercept == TRUE`.
#' @param thresh tolerance level for termination of the algorithm
#' @param return_loss whether to compute and return the loss at each outer
#'   iteration, only added here for debugging purposes.
#' @param ... ignored
#'
#' @return An object of class `'sgdnet'`.
#' @export
#'
#' @examples
#' # Gaussian regression with sparse features
#' x <- Matrix::rsparsematrix(100, 10, density = 0.2)
#' y <- rnorm(100)
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
                           return_loss = FALSE,
                           ...) {

  n_samples <- NROW(x)

  # The internal optimizer uses a different penalty construction than
  # glmnet. Convert lambda vallues to match alpha and beta from scikit-learn.
  alpha_sklearn <- lambda/2*n_samples*(1 - alpha)
  beta_sklearn <- lambda/2*n_samples*alpha

  # Convert sparse x to dgCMatrix class from package Matrix.
  if (is_sparse <- inherits(x, "sparseMatrix")) {
    x <- methods::as(x, "CsparseMatrix")
    x <- methods::as(x, "dgCMatrix")
  } else {
    x <- as.matrix(x)
  }

  y <- as.matrix(y)

  stopifnot(identical(NROW(y), NROW(x)),
            !any(is.na(y)),
            !any(is.na(x)),
            alpha >= 0 && alpha <= 1,
            thresh > 0,
            lambda >= 0,
            is.logical(intercept),
            is.logical(standardize),
            is.logical(return_loss))

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
                  return_loss)

  variable_names <- colnames(x)

  # Organize return values
  a0 <- drop(res$a0)
  names(a0) <- paste0("s", seq_along(a0) - 1L)
  beta <- Matrix::Matrix(res$beta)
  dimnames(beta) <- list(colnames(x), names(a0))

  structure(list(a0 = a0,
                 beta = beta,
                 npasses = res$npasses,
                 losses = if (return_loss) res$losses else NULL),
            class = "sgdnet")
}
