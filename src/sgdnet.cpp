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

#include <RcppEigen.h>
#include "utils.h"
#include "families.h"
#include "prox.h"
#include "saga.h"
#include "math.h"
#include <memory>

//' Setup sgdnet Model Options
//'
//' Collect parameters from `control` and setup storage for coefficients,
//' intercepts, gradients, and more so that we can iterate along the
//' regularization path using warm starts for successive iterations.
//'
//' @param x features
//' @param response the model family
//' @param is_sparse whether x is sparse or not
//' @param control a list of control parameters
//'
//' @return See [SgdnetCpp].
//'
//' @noRd
//' @keywords internal
template <typename T, typename Family>
Rcpp::List SetupSgdnet(T                 x,
                       Family&&          family,
                       const bool        is_sparse,
                       const Rcpp::List& control) {
  using namespace std;
  using namespace sgdnet;

  const bool     fit_intercept    = Rcpp::as<bool>(control["intercept"]);
  const double   elasticnet_mix   = Rcpp::as<double>(control["elasticnet_mix"]);
  vector<double> lambda           = Rcpp::as<vector<double>>(control["lambda"]);
  const unsigned n_lambda         = Rcpp::as<unsigned>(control["n_lambda"]);
  const unsigned n_classes        = Rcpp::as<unsigned>(control["n_classes"]);
  const unsigned n_targets        = Rcpp::as<unsigned>(control["n_targets"]);
  const double   lambda_min_ratio = Rcpp::as<double>(control["lambda_min_ratio"]);
  const bool     standardize      = Rcpp::as<bool>(control["standardize"]);
  const unsigned max_iter         = Rcpp::as<unsigned>(control["max_iter"]);
  const double   tol              = Rcpp::as<double>(control["tol"]);
  const bool     debug            = Rcpp::as<bool>(control["debug"]);

  auto n_samples  = x.rows();
  auto n_features = x.cols();

  // Preprocess features
  vector<double> x_center(n_features);
  vector<double> x_scale(n_features, 1.0);

  if (standardize)
    PreprocessFeatures(x, x_center, x_scale);

  // Transpose x for more efficient access of samples
  AdaptiveTranspose(x);

  // Store null deviance here before processing response
  double null_deviance = family.NullDeviance(fit_intercept);

  // intercept updates are scaled to avoid oscillation
  double intercept_decay = is_sparse ? 0.01 : 1.0;

  family.PreprocessResponse();

  // Compute the lambda sequence
  vector<double> alpha, beta;

  RegularizationPath(lambda,
                     n_lambda,
                     lambda_min_ratio,
                     elasticnet_mix,
                     x,
                     alpha,
                     beta,
                     family);

  // Maximum of sums of squares over samples
  double max_squared_sum = ColNormsMax(x);

  vector<double> step_size =
    family.StepSize(max_squared_sum, alpha, fit_intercept);

  // Check if we need the nontrivial prox
  // TODO(jolars): allow more proximal operators
  sgdnet::SoftThreshold prox;

  // Setup intercept vector
  vector<vector<double>> intercept_archive;

  // Setup weights matrix and weights archive
  vector<double> weights(n_features*n_classes);
  vector<vector<double>> weights_archive;

  // Initialize gradient sum
  vector<double> g_sum(n_features*n_classes);

  // Keep keep track of successes for each penalty
  vector<unsigned> return_codes;

  // Setup a vector of loss vectors to return upon exit
  vector<vector<double>> losses_archive;

  // Keep track of number of iteratios per penalty
  unsigned n_iter = 0;

  // Null deviance on scaled y for computing deviance ratio
  double null_deviance_scaled = family.NullDeviance(fit_intercept);
  vector<double> deviance_ratio;
  deviance_ratio.reserve(n_lambda);

  // Fit the path of penalty values
  for (unsigned lambda_ind = 0; lambda_ind < n_lambda; ++lambda_ind) {
    vector<double> losses;

    Saga(x,
         fit_intercept,
         intercept_decay,
         weights,
         family,
         prox,
         step_size[lambda_ind],
         alpha[lambda_ind],
         beta[lambda_ind],
         g_sum,
         n_samples,
         n_features,
         n_classes,
         max_iter,
         tol,
         n_iter,
         return_codes,
         losses,
         debug);

    double deviance = Deviance(x,
                               weights,
                               n_samples,
                               n_features,
                               n_classes,
                               family);

    deviance_ratio.push_back(1.0 - deviance/null_deviance_scaled);

    // Rescale and store intercepts and weights for the current solution
    Rescale(weights,
            weights_archive,
            family.intercept,
            intercept_archive,
            x_center,
            x_scale,
            family.y_center,
            family.y_scale,
            n_features,
            n_classes,
            fit_intercept);

    if (debug)
      losses_archive.push_back(losses);
  }

  return Rcpp::List::create(
    Rcpp::Named("a0")           = Rcpp::wrap(intercept_archive),
    Rcpp::Named("beta")         = Rcpp::wrap(weights_archive),
    Rcpp::Named("losses")       = Rcpp::wrap(losses_archive),
    Rcpp::Named("npasses")      = n_iter,
    Rcpp::Named("nulldev")      = null_deviance,
    Rcpp::Named("dev.ratio")    = Rcpp::wrap(deviance_ratio),
    Rcpp::Named("lambda")       = Rcpp::wrap(lambda),
    Rcpp::Named("return_codes") = Rcpp::wrap(return_codes)
  );
}

template <typename T>
Rcpp::List SetupFamily(const T&                   x,
                       const std::vector<double>& y,
                       const bool                 is_sparse,
                       const Rcpp::List&          control) {

  auto family_choice = Rcpp::as<std::string>(control["family"]);
  auto n_classes = Rcpp::as<unsigned>(control["n_classes"]);
  auto n_samples = x.rows();

  if (family_choice == "gaussian") {

    sgdnet::Gaussian family(y, n_samples, n_classes);
    return SetupSgdnet(x, std::move(family), is_sparse, control);

  } else if (family_choice == "binomial") {

    sgdnet::Binomial family(y, n_samples, n_classes);
    return SetupSgdnet(x, std::move(family), is_sparse, control);

  } else if (family_choice == "multinomial") {

    sgdnet::Multinomial family(y, n_samples, n_classes);
    return SetupSgdnet(x, std::move(family), is_sparse, control);

  } else {

    return Rcpp::List::create();

  }
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
//'   * return_codes: the convergence result. 0 means that the algorithm
//'     converged, 1 means that `max_iter` was reached before the algorithm
//'     converged.
//
//' @keywords internal
//' @noRd
// [[Rcpp::export]]
Rcpp::List SgdnetDense(const Eigen::MatrixXd&     x,
                       const std::vector<double>& y,
                       const Rcpp::List&          control) {
  return SetupFamily(x, y, false, control);
}

// [[Rcpp::export]]
Rcpp::List SgdnetSparse(const Eigen::SparseMatrix<double>& x,
                        std::vector<double>         y,
                        const Rcpp::List&           control) {
  return SetupFamily(x, y, true, control);
}
