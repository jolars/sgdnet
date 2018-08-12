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
#' @section Model families:
#' Three model families are currently supported: gaussian univariate
#' regression, binomial logistic regression, and multinomial logistic
#' regression. The choice of which is made
#' using the `family` argument. Next follows the objectives of the various
#' model families:
#'
#' *Gaussian univariate regression*:
#'
#' \deqn{
#'   \frac{1}{2n} \sum_{i=1}^n (y_i - \beta_0 - x_i^\mathsf{T} \beta)^2
#'   + \lambda \left( \frac{1 - \alpha}{2} ||\beta||_2^2
#'                    + \alpha||\beta||_1 \right).
#' }{
#'   1/(2n) \sum (y - \beta_0 - x^T \beta)^2
#'   + \lambda [(1 - \alpha)/2 ||\beta||_2^2 + \alpha||\beta||_1].
#' }
#'
#' *Binomial logistic regression*:
#'
#' \deqn{
#'   -\frac1n \sum_{i=1}^n \bigg[y_i (\beta_0 + \beta^\mathsf{T} x_i) -
#'     \log\Big(1 + e^{\beta_0 + \beta^\mathsf{T} x_i}\Big)\bigg]
#'   + \lambda \left( \frac{1 - \alpha}{2} ||\beta||_2^2
#'                    + \alpha||\beta||_1 \right),
#' }{
#'   -1/n \sum_{i=1}^n {y_i (\beta_0 + \beta^T x_i) -
#'     log[1 + exp(\beta_0 + \beta^T x_i)]}
#'   + \lambda [(1 - \alpha)/2 ||\beta||_2^2 + \alpha||\beta||_1],
#' }
#' where \eqn{y_i \in \{0, 1\}}{y ~ {0, 1}}.
#'
#' *Multinomial logistic regression*:
#' \deqn{
#'  -\bigg\{\frac1n \sum_{i=1}^n \Big[\sum_{k=1}^m y_{i_k} (\beta_{0_k} + x_i^\mathsf{T} \beta_k) -
#'  \log \sum_{k=1}^m e^{\beta_{0_k}+x_i^\mathsf{T} \beta_k}\Big]\bigg\}
#'  + \lambda \left( \frac{1 - \alpha}{2}||\beta||_F^2 + \alpha \sum_{j=1}^p ||\beta_j||_q \right),
#' }{
#'  -{1\n \sum_{i=1}^n [\sum_{k=1}^m y_{i_k} (\beta_{0_k} + x_i^T \beta_k) -
#'  \log \sum_{k=1}^m exp(\beta_{0_k}+x_i^T \beta_k)]}
#'  + \lambda ((1 - \alpha)/2||\beta||_F^2 + \alpha \sum_{j=1}^p ||\beta_j||_q),
#' }
#' where \eqn{q \in {1, 2}}{q = {1,2}} invokes the standard lasso and 2 the
#' group lasso penalty respectively, \eqn{F} indicates the Frobenius norm,
#' and \eqn{p} is the number of classes.
#'
#' *Multivariate gaussian regression*:
#' \deqn{
#'   \frac{1}{2n} ||\mathbf{Y} -\mathbf{B}_0\mathbf{1} - \mathbf{B} \mathbf{X}||^2_F
#'   + \lambda \left((1 - \alpha)/2||\mathbf{B}||_F^2 + \alpha ||\mathbf{B}||_{12}\right),
#' }{
#'   1/(2n) ||Y - B_01 - BX||_F^2 + \lambda (1 - \alpha)/2||B||_F^2 + \alpha ||B||_12),
#' }
#' where \eqn{\mathbf{1}}{1} is a vector of all zeros, \eqn{\mathbf{B}}{B} is a
#' matrix of coefficients, and \eqn{||\dot||_{12}}{||.||_12} is the mixed
#' \eqn{\ell_{1/2}}{L1/2} norm. Note, also, that Y is a matrix
#' of responses in this form.
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
#' @section Relationship with glmnet:
#' **sgdnet** is modeled to resemble [glmnet][glmnet::glmnet-package] closely so that users
#' can expect to receive more or less equivalent output regardless of whether
#' [sgdnet()] or [glmnet::glmnet()] is called. Nevertheless, there are a
#' few instances where we have decided to diverge from the behavior of
#' [glmnet][glmnet::glmnet-package]:
#' \itemize{
#'   \item When the ridge penalty is used (`alpha = 0`), and a regularization
#'         path (\eqn{\lambda~s}{lambdas}) is automatically generated,
#'         [glmnet::glmnet()] fits the null model as the start of the path
#'         (as if \eqn{\lambda = \infty}{lambda = inf})
#'         even though the first \eqn{\lambda}{lambda} reported actually
#'         doesn't yield this fit. In [sgdnet][sgdnet-package], we have opted
#'         to fit the model so that it is true to the path that is returned.
#' }
#'
#' @param x input matrix
#' @param y response variable
#' @param family reponse type, one of `'gaussian'`, `'binomial'`,
#'   `'multinomial'`, or `'mgaussian'`. See **Supported families** for details.
#' @param alpha elastic net mixing parameter
#' @param nlambda number of penalties in the regualrization path
#' @param lambda.min.ratio the ratio between `lambda_max` (the smallest
#'   penalty at which the solution is completely sparse) and the smallest
#'   lambda value on the path. See **Regularization Path** for details.
#' @param lambda regularization strength
#' @param intercept whether to fit an intercept or not
#' @param maxit maximum number of effective passes (epochs)
#' @param standardize whether to standardize `x` or not
#' @param thresh tolerance level for termination of the algorithm. The
#'   algorithm terminates when
#'   \deqn{
#'     \frac{|\beta^{(t)}
#'     - \beta^{(t-1)}|{\infty}}{|\beta^{(t)}|{\infty}} < \mathrm{thresh}
#'   }{
#'     max(change in weights)/max(weights) < thresh.
#'   }
#' @param standardize.response whether `y` should be standardized for
#'   `family = "mgaussian"`
#' @param ... ignored
#'
#' @return An object of class `'sgdnet'` with the following items:
#' \item{`a0`}{the intercept}
#' \item{`beta`}{the coefficients stored in sparse matrix format
#'               "dgCMatrix". For the multivariate families, this is a
#'               list with one matrix of coefficients for each response or
#'               class.}
#' \item{`nulldev`}{the deviance of the null (intercept-only model)}
#' \item{`dev.ratio`}{the fraction of deviance explained, where the deviance
#'                    is two times the difference in loglikelihood between the
#'                    saturated model and the null model}
#' \item{`df`}{the number of nozero coefficients along the
#'             regularization path. For `family = "multinomial"`,
#'             this is the number of variables with
#'             a nonzero coefficient for any class.}
#' \item{`dfmat`}{a matrix of the number of nonzero coefficients for
#'                any class (only available for multivariate models)}
#' \item{`alpha`}{elastic net mixing parameter. See the description
#'                of the arguments.}
#' \item{`lambda`}{the sequence of lambda values scaled to the
#'                 original scale of the input data.}
#' \item{`nobs`}{number of observations}
#' \item{`npasses`}{accumulated number of outer iterations (epochs)
#'                  for the entire regularization path}
#' \item{`offset`}{a logical indicating whether an offset was used}
#' \item{`grouped`}{a logical indicating if a group lasso penalty was used}
#' \item{`call`}{the call that generated this fit}
#'
#' @seealso [predict.sgdnet()], [plot.sgdnet()], [coef.sgdnet()],
#'   [sgdnet-package()]
#'
#' @export
#'
#' @examples
#' # Gaussian regression with sparse features with ridge penalty
#' fit <- sgdnet(abalone$x, abalone$y, alpha = 0)
#'
#' # Binomial logistic regression with elastic net penalty, no intercept
#' binom_fit <- sgdnet(heart$x,
#'                     heart$y,
#'                     family = "binomial",
#'                     alpha = 0.5,
#'                     intercept = FALSE)
#'
#' # Multinomial logistic regression with lasso
#' multinom_fit <- sgdnet(wine$x, wine$y, family = "multinomial")
#'
#' # Multivariate gaussian regression
#' mgaussian_fit <- sgdnet(student$x, student$y, family = "mgaussian")
sgdnet <- function(x, ...) UseMethod("sgdnet")

#' @export
#' @rdname sgdnet
sgdnet.default <- function(x,
                           y,
                           family = c("gaussian",
                                      "binomial",
                                      "multinomial",
                                      "mgaussian"),
                           alpha = 1,
                           nlambda = 100,
                           lambda.min.ratio =
                             if (NROW(x) < NCOL(x)) 0.01 else 0.0001,
                           lambda = NULL,
                           maxit = 1000,
                           standardize = TRUE,
                           intercept = TRUE,
                           thresh = 0.001,
                           standardize.response = FALSE,
                           ...) {

  # Collect the call so we can use it in update() later on
  ocall <- match.call()

  # Collect sgdnet-specific options for debugging and more
  debug <- getOption("sgdnet.debug")

  n_samples <- NROW(x)
  n_features <- NCOL(x)
  n_targets <- NCOL(y)

  stopifnot(is.logical(intercept),
            is.logical(standardize),
            is.logical(debug))

  if (NROW(y) != NROW(x))
    stop("the number of samples in 'x' and 'y' must match")

  if (NROW(y) == 0)
    stop("the response (y) is empty.")

  if (NROW(x) == 0)
    stop("the predictor matrix (x) is empty.")

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
  class_names    <- NULL

  if (is.null(variable_names))
    variable_names <- paste0("V", seq_len(NCOL(x)))
  if (is.null(response_names))
    response_names <- paste0("y", seq_len(NCOL(y)))

  if (is.null(lambda) || is_false(lambda))
    lambda <- double(0L)
  else
    nlambda <- length(lambda)

  if (nlambda == 0)
    stop("lambda path cannot be of zero length.")

  if (alpha < 0 || alpha > 1)
    stop("elastic net mixing parameter (alpha) must be in [0, 1].")

  if (any(lambda < 0))
    stop("penalty strengths (lambdas) must be positive.")

  if (any(is.na(y)) || any(is.na(x)))
    stop("NA values are not allowed.")

  if (thresh < 0)
    stop("threshold for stopping criteria cannot be negative.")

  if (maxit <= 0)
    stop("maximum number of iterations cannot be negative or zero.")

  # TODO(jolars): implement group lasso penalty for multinomial model
  type.multinomial <- "ungrouped"
  switch(
    type.multinomial,
    ungrouped = {
      grouped <- FALSE
    }
  )

  # Setup reponse type options and assert appropriate input
  family <- match.arg(family)

  switch(
    family,
    gaussian = {
      if (n_targets > 1)
        stop("response for Gaussian regression must be one-dimensional.")

      if (!is.numeric(y))
        stop("non-numeric response.")

      n_classes <- 1L
      y <- as.numeric(y)
    },
    binomial = {
      if (length(unique(y)) > 2)
        stop("more than two classes in response. Are you looking for family = 'multinomial'?")

      if (length(unique(y)) == 1)
        stop("only one class in response.")

      y_table <- table(y)
      min_class <- min(y_table)
      n_classes <- 1L

      if (min_class <= 1)
        stop("one class only has ", min_class, " observations.")

      class_names <- names(y_table)

      # Transform response to {-1, 1}, which is used internally
      y <- as.numeric(as.factor(y)) - 1
    },
    multinomial = {
      y <- droplevels(as.factor(y))

      y_table <- table(y)
      min_class <- min(y_table)
      class_names <- names(y_table)
      n_classes <- length(y_table)

      if (n_classes == 2)
        stop("only two classes in response. Are you looking for family = 'binomial'?")

      if (n_classes == 1)
        stop("only one class in response.")

      if (min_class <= 1)
        stop("one class only has ", min_class, " observations.")

      y <- as.numeric(y) - 1
    },
    mgaussian = {
      if (n_targets == 1)
        stop("response for multivariate Gaussian regression must not be one-dimensional; try family = 'gaussian'.")

      if (!is.numeric(y))
        stop("non-numeric response.")

      class_names <- colnames(y)
      grouped <- TRUE

      n_classes <- n_targets
    }
  )

  # TODO(jolars): implement offset
  offset <- FALSE

  y <- as.matrix(y)

  control <- list(debug = debug,
                  elasticnet_mix = alpha,
                  family = family,
                  intercept = intercept,
                  is_sparse = is_sparse,
                  lambda = lambda,
                  lambda_min_ratio = lambda.min.ratio,
                  max_iter = maxit,
                  n_lambda = nlambda,
                  n_classes = n_classes,
                  standardize = standardize,
                  standardize_response = standardize.response,
                  tol = thresh,
                  type_multinomial = type.multinomial)

  # Fit the model by calling the Rcpp routine.
  if (is_sparse) {
    res <- SgdnetSparse(x, y, control)
  } else {
    res <- SgdnetDense(x, y, control)
  }

  lambda <- res$lambda
  n_penalties <- length(lambda)
  path_names <- paste0("s", seq_len(n_penalties) - 1L)

  if (family %in% c("gaussian", "binomial")) {
    # NOTE(jolars): I would rather not to this, i.e. have different outputs
    # depending on family, but this makes it equivalent to glmnet output
    a0 <- unlist(res$a0)
    names(a0) <- path_names
    beta <- Matrix::Matrix(unlist(res$beta),
                           nrow = n_features,
                           ncol = n_penalties,
                           dimnames = list(variable_names, path_names),
                           sparse = TRUE)

    df <- Matrix::colSums(beta != 0)

  } else if (family %in% c("multinomial", "mgaussian")) {
    a0 <- matrix(unlist(res$a0, use.names = FALSE), ncol = n_penalties)
    colnames(a0) <- path_names
    rownames(a0) <- class_names
    tmp <- matrix(unlist(res$beta), nrow = n_classes)
    beta <- vector("list", n_classes)
    names(beta) <- switch(family,
                          mgaussian = response_names,
                          multinomial = class_names)
    for (i in seq_len(n_classes)) {
      beta[[i]] <- Matrix::Matrix(tmp[i, ],
                                  nrow = n_features,
                                  ncol = n_penalties,
                                  dimnames = list(variable_names, path_names),
                                  sparse = TRUE)
    }

    df <- Matrix::colSums(Reduce("+", beta) != 0)

    dfmat <- lapply(beta, function(x) apply(x, 2, function(x) sum(abs(x) > 0)))
    dfmat <- do.call(rbind, dfmat)
  }

  # make sure that intercepts for the multinomial family sum to 0
  if (family == "multinomial")
    a0 <- t(t(a0) - colMeans(a0))

  out <- structure(list(a0 = a0,
                        beta = beta,
                        lambda = lambda,
                        dev.ratio = res$dev.ratio,
                        df = df,
                        nulldev = res$nulldev,
                        npasses = res$npasses,
                        alpha = alpha,
                        offset = offset,
                        classnames = class_names,
                        grouped = grouped,
                        call = ocall,
                        nobs = n_samples),
                   class = c(paste0("sgdnet_", family), "sgdnet"))

  if (family %in% c("multinomial", "mgaussian"))
    out$dfmat <- dfmat

  if (debug)
    attr(out, "diagnostics") <- list(loss = res$losses)
  out
}
