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

#' *k*-fold Cross Validation for **sgdnet**
#'
#' @inheritParams sgdnet
#' @param alpha elastic net mixing parameter; vectors of values are allowed
#'  (unlike in [sgdnet()])
#' @param nfolds number of folds (k)
#' @param foldid a vector of fold identities
#' @param type.measure the type of error
#' @param ... arguments passed on to [sgdnet()]
#'
#' @return An object of class `'cv_sgdnet'` with the following items:
#' \tabular{ll}{
#'   `alpha`      \tab the elastic net mixing parameter used\cr
#'   `lambda`     \tab a list of lambda values of the same length as `alpha`\cr
#'   `cv_summary` \tab a `data.frame` summarizing the prediction error across
#'                     the regularization path with columns `alpha`, `lambda`,
#'                     `mean`, `sd`, `ci_lo`, `ci_up`\cr
#'   `cv_raw`     \tab the raw cross-validation scores as a list of the
#'                     same length as `alpha`, each item a `matrix` with
#'                     the error for each fold as a row and each value of
#'                     `lambda` in columns.\cr
#'   `name`       \tab the type of prediction error used\cr
#'   `fit`        \tab a fit from [sgdnet()] to the full data set based on the
#'                     `alpha` with the best cross-validation score\cr
#'   `alpha_min`  \tab the `alpha` corresponding to the fit with the best
#'                     cross-validation performance\cr
#'   `lambda_min` \tab the `lambda` corresponding to the fit with the best
#'                     cross-validation performance\cr
#'   `lambda_1se` \tab the largest `lambda` with a cross-validation performance
#'                     within one standard deviation of the one
#'                     coresponding to `lambda_min`
#' }
#'
#' @seealso [sgdnet()], [predict.cv_sgdnet()], [plot.cv_sgdnet()]
#'
#' @export
#'
#' @examples
#' set.seed(1)
#' n <- nrow(heart$x)
#' train_ind <- sample(n, floor(0.8*n))
#' cv_fit <- cv_sgdnet(heart$x[train_ind, ],
#'                     heart$y[train_ind],
#'                     family = "binomial",
#'                     nfolds = 7,
#'                     alpha = c(0, 1))
#' plot(cv_fit)
#' predict(cv_fit, heart$x[-train_ind, ], s = "lambda_min")
cv_sgdnet <- function(x,
                      y,
                      alpha = 1,
                      lambda = NULL,
                      nfolds = 10,
                      foldid = NULL,
                      type.measure = c("deviance",
                                       "mse",
                                       "mae",
                                       "class",
                                       "auc"),
                      ...) {

  stopifnot(nfolds > 2, is.numeric(alpha), length(alpha) > 0)

  type.measure <- match.arg(type.measure)

  x <- as.matrix(x)
  y <- as.matrix(y)

  n_samples <- nrow(x)
  n_features <- ncol(x)
  n_alpha <- length(alpha)

  if (!is.null(nfolds)) {
    if (nfolds > n_samples)
      stop("you cannot have more folds than samples.")
  }

  if (is.list(lambda)) {

    if (n_alpha != length(lambda))
      stop("the length of the lambda list needs to match the number of alpha.")

  } else if (is.vector(lambda)) {

    if (n_alpha > 1)
      stop("you need a list of lambdas (or have it set at NULL) when you have multiple alphas.")

    lambda <- list(lambda)

  } else if (is.null(lambda)) {

    lambda <- replicate(n_alpha, c(NULL))

  }

  fits <- vector("list", n_alpha)
  for (i in seq_along(alpha))
    fits[[i]] <- sgdnet(x, y, lambda = lambda[[i]], alpha = alpha[i], ...)

  lambda <- lapply(fits, "[[", "lambda")

  nlambda <- vapply(fits, function(x) length(x$lambda), double(1))

  if (is.null(foldid))
    foldid <- as.numeric(cut(sample(n_samples), nfolds))
  else {
    if (length(foldid) != n_samples)
      stop("the length of `foldid` must match the number of samples")
    nfolds <- length(unique(foldid))
  }

  cv_raw <- vector("list", n_alpha)

  for (i in seq_along(alpha)) {
    cv_raw[[i]] <- matrix(NA, nfolds, nlambda[i])

    for (j in seq_len(nfolds)) {
      train_ind <- j == foldid
      test_ind <- !train_ind

      x_train <- x[train_ind, , drop = FALSE]
      y_train <- y[train_ind, , drop = FALSE]

      x_test <- x[test_ind, , drop = FALSE]
      y_test <- y[test_ind, , drop = FALSE]

      fit <- sgdnet(x_train,
                    y_train,
                    lambda = lambda[[i]],
                    alpha = alpha[i],
                    ...)

      cv_raw[[i]][j, ] <-
        score(fit, x_test, y_test, type.measure = type.measure)
    }
  }

  alpha_names <- paste0("alpha_", alpha)

  cv_summary <- matrix(NA, sum(nlambda), 6)

  for (i in seq_len(n_alpha)) {
    ind <- ((i - 1)*nlambda[i] + 1):(i*nlambda[i])
    cv_summary[ind, 1]   <- rep.int(alpha[[i]], nlambda[i])
    cv_summary[ind, 2]   <- lambda[[i]]
    cv_summary[ind, 3:6] <- summarize_cv_raw(cv_raw[[i]])
  }

  cv_summary <- as.data.frame(cv_summary)

  colnames(cv_summary) <- c("alpha", "lambda", "mean", "sd", "ci_lo", "ci_up")

  optima <- do.call(rbind, by(cv_summary,
                              as.factor(cv_summary$alpha),
                              find_optimum,
                              simplify = FALSE))

  ind_min <- which.min(optima$error_min)

  alpha_min  <- optima$alpha_min[ind_min]
  lambda_min <- optima$lambda_min[ind_min]
  lambda_1se <- optima$lambda_1se[ind_min]

  name <- switch(
    type.measure,
    deviance = {
      if (inherits(fit, c("sgdnet_gaussian", "sgdnet_mgaussian")))
        "Mean-Squared Error"
      else if (inherits(fit, "sgdnet_binomial"))
        "Binomial Deviance"
      else if (inherits(fit, "sgdnet_multinomial"))
        "Multnomial Deviance"
    },
    mse = "Mean-Squared Error",
    mae = "Mean Absolute Error",
    class = "Misclassification Error",
    auc = "AUC"
  )

  structure(list(alpha = alpha,
                 lambda = lambda,
                 cv_summary = cv_summary,
                 cv_raw = cv_raw,
                 name = name,
                 fit = fits[[ind_min]],
                 alpha_min = alpha_min,
                 lambda_min = lambda_min,
                 lambda_1se = lambda_1se),
            class = "cv_sgdnet")
}

#' Find Optimal Lambda
#'
#' @param cv_summary A summary of cross-validation results computed from
#'   summarize_cv_raw().
#'
#' @return A list of values indicating where the optimal lambda are.
#'
#' @keywords internal
#' @noRd
find_optimum <- function(cv_summary) {
  alpha   <- cv_summary[, 1]
  lambda  <- cv_summary[, 2]
  cv_mean <- cv_summary[, 3]
  cv_sd   <- cv_summary[, 4]

  ind_min    <- which.min(cv_mean)
  alpha_min  <- alpha[ind_min]
  lambda_min <- lambda[ind_min]

  within_1se <- cv_mean <= cv_mean[ind_min] + cv_sd[ind_min]
  lambda_1se <- max(lambda[within_1se])

  data.frame(alpha_min  = alpha_min,
             lambda_min = lambda_min,
             lambda_1se = lambda_1se,
             error_min  = cv_mean[ind_min])
}

#' Summarize Raw Data From Cross-Validation
#'
#' @param cv_raw a matrix of raw cross-validation measures
#'
#' @return A matrix of means, standard deviations, and confidence limits
#'   for the aggregated cross-validation results.
#'
#' @keywords internal
#' @noRd
summarize_cv_raw <- function(cv_raw) {
  cv_bar <- colMeans(cv_raw, TRUE)
  cv_sd <- col_sd(cv_raw)
  ci_lo <- cv_bar - cv_sd
  ci_up <- cv_bar + cv_sd
  matrix(c(cv_bar, cv_sd, ci_lo, ci_up), nrow = length(cv_bar), ncol = 4)
}


