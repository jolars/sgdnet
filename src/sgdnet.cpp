// sgdnet: Penalized Generalized Linear Models with Stochastic Gradient Descent
// Copyright (C) 2018  Johan Larsson
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

// This code is part translation from the Python package scikit-learn,
// which comes with the following copyright notice:
//
// Copyright (c) 2007–2018 The scikit-learn developers.
// All rights reserved.
//
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// a. Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// b. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
// c. Neither the name of the Scikit-learn Developers  nor the names of
// its contributors may be used to endorse or promote products
// derived from this software without specific prior written
// permission.
//
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
// DAMAGE.

// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include "utils.h"
#include "families.h"
#include "prox.h"
#include "saga.h"
#include <memory>

//' Rescale weights and intercept before returning these to user
//'
//' Currently no processing, and therefore no rescaling, is done
//' when the intercept is fit. If x is sparse.
//'
//' @param weights weights
//' @param intercept intercept
//' @param x_center the offset (mean) used to possibly have centered x
//' @param x_scale the scaling that was applied to x
//' @param y_center the offset (mean) that y was offset with
//' @param n_features the number of features
//' @param fit_intercept whether to fit the intercept
//' @param is_sparse whether the features are sparse
//'
//' @return `weights` and `intercept` are rescaled.
//'
//' @noRd
//' @keywords internal
void Rescale(arma::cube&         weights,
             arma::mat&          intercept,
             const arma::rowvec& x_center,
             const arma::rowvec& x_scale,
             const arma::rowvec& y_center,
             const arma::rowvec& y_scale,
             const arma::uword   n_features,
             const arma::uword   fit_intercept) {

  if (fit_intercept) {
    for (arma::uword i = 0; i < n_features; ++i) {
      if (x_scale(i) != 0.0) {
        for (arma::uword j = 0; j < y_scale.n_elem; ++j)
          weights.tube(i, j) *= y_scale(j)/x_scale(i);
      }
    }

    for (arma::uword i = 0; i < weights.n_slices; ++i) {
      intercept.row(i) =
        intercept.row(i)*y_scale + y_center - x_center*weights.slice(i);
    }
  }
}

//' Compute Regularization Path
//'
//' This function computes the regularization path as in glmnet so that
//' the first solution is the null solution (if elasticnet_mix != 0).
template <typename T>
void RegularizationPath(arma::vec&          lambda,
                        const arma::uword   n_lambda,
                        const double        lambda_min_ratio,
                        const double        elasticnet_mix,
                        const T&            x,
                        const arma::mat&    y,
                        const arma::uword   n_samples,
                        arma::vec&          alpha,
                        arma::vec&          beta) {

  if (lambda.is_empty()) {

    double lambda_max = arma::abs(y.t() * x).max() / n_samples;
    // Cap elasticnet_mix (alpha in glmnet) to 0.001
    lambda_max /= std::max(elasticnet_mix, 0.001);

    lambda = arma::exp(arma::linspace<arma::vec>(
      std::log(lambda_max), std::log(lambda_max*lambda_min_ratio), n_lambda));
  }

  // The algorithm uses a different penalty construction than
  // glmnet, so convert lambda values to match alpha and beta from scikit-learn.

  // Scaled L2 penalty
  alpha = lambda*(1 - elasticnet_mix)/2;
  // Scaled L1 penalty
  beta = lambda*elasticnet_mix;
}

//' Adapative transposing of feature matrix
//'
//' For sparse matrices, armadillo does not (yet?) have a inplace
//' transpose method, so we overload for sparse and dense matrices,
//' transposing inplace when x is dense.
//'
//' @param x a sparse or dense matrix
//'
//' @return x transposed.
//'
//' @keywords internal
//' @noRd
void AdaptiveTranspose(arma::sp_mat& x) {
  x = x.t();
}

void AdaptiveTranspose(arma::mat& x) {
  arma::inplace_trans(x);
}


//' Setup sgdnet Model Options
//'
//' Collect parameters from `control` and setup storage for coefficients,
//' intercepts, gradients, and more so that we can iterate along the
//' regularization path using warm starts for successive iterations.
//'
//' @param x features
//' @param y response
//' @param is_sparse whether x is sparse or not
//' @param control a list of control parameters
//'
//' @return See [SgdnetCpp].
//'
//' @noRd
//' @keywords internal
template <typename T>
Rcpp::List SetupSgdnet(T&                x,
                       arma::mat         y,
                       bool              is_sparse,
                       const Rcpp::List& control) {

  std::string  family_choice    = Rcpp::as<std::string>(control["family"]);
  bool         fit_intercept    = Rcpp::as<bool>(control["intercept"]);
  double       elasticnet_mix   = Rcpp::as<double>(control["elasticnet_mix"]);
  arma::vec    lambda           = Rcpp::as<arma::vec>(control["lambda"]);
  arma::uword  n_lambda         = Rcpp::as<arma::uword>(control["n_lambda"]);
  double       lambda_min_ratio = Rcpp::as<double>(control["lambda_min_ratio"]);
  bool         normalize        = Rcpp::as<bool>(control["normalize"]);
  arma::uword  max_iter         = Rcpp::as<arma::uword>(control["max_iter"]);
  double       tol              = Rcpp::as<double>(control["tol"]);
  bool         debug            = Rcpp::as<bool>(control["debug"]);

  arma::uword n_samples   = x.n_rows;
  arma::uword n_features  = x.n_cols;
  arma::uword n_targets   = y.n_cols;

  // Preprocess features
  arma::rowvec x_center(n_features);
  arma::rowvec x_scale(n_features);

  PreprocessFeatures(x,
                     normalize,
                     fit_intercept,
                     x_center,
                     x_scale,
                     is_sparse,
                     n_features);

  double intercept_decay = is_sparse ? 0.01 : 1.0;

  // Setup family-specific options
  sgdnet::FamilyFactory family_factory;
  std::unique_ptr<sgdnet::Family> family =
    family_factory.NewFamily(family_choice);

  arma::uword n_classes = family->NClasses(y);

  // Preprocess response
  arma::rowvec y_center(n_targets);
  arma::rowvec y_scale(n_targets);

  // Compute the lambda sequence
  arma::vec alpha;
  arma::vec beta;

  family->PreprocessResponse(y, y_center, y_scale, fit_intercept);

  RegularizationPath(lambda,
                     n_lambda,
                     lambda_min_ratio,
                     elasticnet_mix,
                     x,
                     y,
                     n_samples,
                     alpha,
                     beta);

  // FIXME(jolars): This needs to be handled appropriately when we extend to
  // multivariate regression.
  lambda *= y_scale(0);

  arma::uword n_penalties = lambda.n_elem;

  // Transpose x for more efficient access of samples
  AdaptiveTranspose(x);

  // Maximum of sums of squares over samples
  double max_squared_sum = ColNormsMax(x);

  arma::vec step_size = family->StepSize(max_squared_sum,
                                         alpha,
                                         fit_intercept,
                                         n_samples);

  // Check if we need the nontrivial prox
  // TODO(jolars): allow more proximal operators
  sgdnet::ProxFactory prox_factory;
  std::string prox_choice = "soft_threshold";
  std::unique_ptr<sgdnet::Prox> prox = prox_factory.NewProx(prox_choice);

  // Setup intercept vector
  arma::rowvec intercept(n_classes, arma::fill::zeros);
  arma::mat intercept_archive(n_penalties, n_classes);

  // Setup weights matrix and weights archive
  arma::mat weights(n_features, n_classes, arma::fill::zeros);
  arma::cube weights_archive(n_features, n_classes, n_penalties);

  // Sum of gradients for weights
  arma::mat sum_gradient(weights);

  // Sum of gradients for intercept
  arma::rowvec intercept_sum_gradient(n_classes, arma::fill::zeros);

  // Gradient memory
  arma::mat gradient_memory(n_samples, n_classes, arma::fill::zeros);

  // Keep track of the number of as well as which samples are seen
  arma::uvec seen(n_samples, arma::fill::zeros);
  arma::uword n_seen = 0;

  // Keep keep track of successes for each penalty
  std::vector<int> return_codes;
  return_codes.reserve(n_penalties);
  arma::uword return_code;

  // Setup a field of losses to return upon exit
  arma::field<arma::vec> losses_archive(n_penalties);
  std::vector<double> losses;
  losses.reserve(n_penalties);

  // Setup a vector to compute deviance at each iteration
  std::vector<double> deviance_ratio;
  deviance_ratio.reserve(n_penalties);
  arma::mat prediction(n_samples, n_classes);

  double null_deviance =
    arma::accu(arma::sum(arma::square(y)) % arma::square(y_scale));

  // Keep track of number of iteratios per penalty
  arma::uword n_iter = 0;

  // Fit the path of penalty values
  for (arma::uword penalty_ind = 0; penalty_ind < n_penalties; ++penalty_ind) {
    Saga(x,
         y,
         weights,
         fit_intercept,
         intercept,
         intercept_decay,
         intercept_sum_gradient,
         family,
         prox,
         step_size(penalty_ind),
         alpha(penalty_ind),
         beta(penalty_ind),
         sum_gradient,
         gradient_memory,
         seen,
         n_seen,
         n_samples,
         n_features,
         n_classes,
         is_sparse,
         max_iter,
         tol,
         n_iter,
         return_code,
         losses,
         debug);

    // Compute deviance
    prediction = x.t() * weights;
    deviance_ratio.push_back(family->Deviance(prediction, y));

    // Store intercepts and weights for the current solution
    weights_archive.slice(penalty_ind) = weights;
    intercept_archive.row(penalty_ind) = intercept;
    return_codes.push_back(return_code);

    if (debug) {
      // Store losses
      losses_archive(penalty_ind) = arma::conv_to<arma::vec>::from(losses);
      // Reset loss
      losses.clear();
    }
  }

  // Rescale intercept and weights back to original scale
  Rescale(weights_archive,
          intercept_archive,
          x_center,
          x_scale,
          y_center,
          y_scale,
          n_features,
          fit_intercept);

  return Rcpp::List::create(
    Rcpp::Named("a0") = Rcpp::wrap(intercept_archive),
    Rcpp::Named("beta") = Rcpp::wrap(weights_archive),
    Rcpp::Named("losses") = Rcpp::wrap(losses_archive),
    Rcpp::Named("npasses") = n_iter,
    Rcpp::Named("nulldev") = null_deviance,
    Rcpp::Named("dev.ratio") = Rcpp::wrap(deviance_ratio),
    Rcpp::Named("lambda") = Rcpp::wrap(lambda),
    Rcpp::Named("return_codes") = Rcpp::wrap(return_codes)
  );
}

//' Fit a Model with sgdnet
//'
//' This main use of this function is calling the templated SetupSgdnet()
//' so that the dense and sparse implementations are compiled and
//' called appropriately. The control parameters in `control` are just
//' passed along.
//'
//' @param x_in feature matrix
//' @param y response matrix
//' @param control a list of control parameters
//'
//' @return A list of
//'   * ao: the intercept,
//'   * beta: the weights,
//'   * losses: the loss at each outer iteration per fit,
//'   * npasses: the number of effective passes (epochs) accumulated over,
//'     all lambda values, and
//'   * return_codes: the convergence result. 0 mean that the algorithm
//'     converged, 1 means that `max_iter` was reached before the algorithm
//'     converged.
//'
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List SgdnetCpp(SEXP              x_in,
                     arma::mat&        y,
                     const Rcpp::List& control) {

  bool is_sparse = Rcpp::as<bool>(control["is_sparse"]);

  if (is_sparse) {
    arma::sp_mat x = Rcpp::as<arma::sp_mat>(x_in);
    return SetupSgdnet(x, y, is_sparse, control);
  } else {
    arma::mat x = Rcpp::as<arma::mat>(x_in);
    return SetupSgdnet(x, y, is_sparse, control);
  }
}

