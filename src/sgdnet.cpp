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
#include "math.h"
#include <memory>
//#include <gperftools/profiler.h>

//' Deviance
//'
//' Computes the deviance of the model given by `weights` and `intercept`.
//'
//' @param x a feature matrix (dense or sparse)
//' @param y a response vector
//' @param weights a vector of coefficients
//' @param intercept an intercept vector
//' @param is_sparse whether x is sparse
//' @param n_samples the number of samples
//' @param n_feature the number of features
//' @param n_classes the number of classes
//'
//' @return Returns the deviance.
//'
//' @noRd
//' @keywords internal
template <typename T>
double Deviance(const T&                         x,
                const std::vector<double>&       y,
                const std::vector<double>&       weights,
                const std::vector<double>&       intercept,
                const bool                       is_sparse,
                const std::size_t                n_samples,
                const std::size_t                n_features,
                const std::size_t                n_classes,
                std::unique_ptr<sgdnet::Family>& family) {

  double loss = 0.0;

  std::vector<std::size_t> nonzero_indices = Nonzeros(x.col(0));

  for (std::size_t sample_ind = 0; sample_ind < n_samples; ++sample_ind) {
    if (is_sparse && sample_ind > 0)
      nonzero_indices = Nonzeros(x.col(sample_ind));

    for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind) {
      auto x_itr = x.begin_col(sample_ind);
      double inner_product = 0.0;
      for (const auto& feature_ind : nonzero_indices) {
        inner_product += (*x_itr)*weights[feature_ind*n_classes + class_ind];
        ++x_itr;
      }
      loss += family->Loss(inner_product + intercept[class_ind],
                           y[sample_ind*n_classes + class_ind]);
    }
  }
  return 2.0 * loss;
}

//' Rescale weights and intercept before returning these to user
//'
//' Currently no processing, and therefore no rescaling, is done
//' when the intercept is fit. If x is sparse.
//'
//' @param weights weights
//' @param weights_archive storage for weights
//' @param intercept intercept
//' @param intercept_archive storage for intercepts on a per-penalty basis
//' @param x_center the offset (mean) used to possibly have centered x
//' @param x_scale the scaling that was applied to x
//' @param y_center the offset (mean) that y was offset with
//' @param y_scale scaling for y
//' @param n_features the number of features
//' @param n_classes number of classes
//' @param fit_intercept whether to fit the intercept
//'
//' @return `weights` and `intercept` are rescaled and stored in weights_archive
//'   and intercept_archive.
//'
//' @noRd
void Rescale(std::vector<double>                 weights,
             std::vector< std::vector<double> >& weights_archive,
             std::vector<double>                 intercept,
             std::vector< std::vector<double> >& intercept_archive,
             const std::vector<double>&          x_center,
             const std::vector<double>&          x_scale,
             const std::vector<double>&          y_center,
             const std::vector<double>&          y_scale,
             const std::size_t                   n_features,
             const std::size_t                   n_classes,
             const bool                          fit_intercept) {

  if (fit_intercept) {
    long double x_scale_prod = 0.0;
    for (std::size_t feature_ind = 0; feature_ind < n_features; ++feature_ind) {
      if (x_scale[feature_ind] != 0.0) {
        for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind) {
          weights[feature_ind*n_classes + class_ind] *=
            y_scale[class_ind]/x_scale[feature_ind];
          x_scale_prod +=
            x_center[feature_ind]*weights[feature_ind*n_classes + class_ind];
        }
      }
    }

    for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind)
      intercept[class_ind] = intercept[class_ind]
                             * y_scale[class_ind]
                             + y_center[class_ind]
                             - x_scale_prod;
  }
  weights_archive.push_back(weights);
  intercept_archive.push_back(intercept);
}

//' Compute Regularization Path
//'
//' This function computes the regularization path as in glmnet so that
//' the first solution is the null solution (if elasticnet_mix != 0).
//'
//' @param lambda lambda values in input -- empty by default
//' @param n_lambda required number of penalties
//' @param lambda_min_ratio smallest lambda_min_ratio
//' @param elasticnet_mix ratio of l1 penalty to l2. Same as alpha in glmnet.
//' @param x feature matrix
//' @param y response vector
//' @param n_samples number of samples
//' @param alpha l2-penalty
//' @param beta l1-penalty
//' @param y_scale scale of y, used only to return lambda values to same
//'   scale as in glmnet
//'
//' @return lambda, alpha and beta are updated.
//'
//' @noRd
template <typename T>
void RegularizationPath(std::vector<double>&       lambda,
                        const std::size_t          n_lambda,
                        const double               lambda_min_ratio,
                        const double               elasticnet_mix,
                        const T&                   x,
                        const std::vector<double>& y,
                        const std::size_t          n_samples,
                        std::vector<double>&       alpha,
                        std::vector<double>&       beta,
                        const std::vector<double>& y_scale) {

  double alpha_ratio = 2.0*(1.0 - elasticnet_mix);
  double beta_ratio = elasticnet_mix;
  double scaling = alpha_ratio + beta_ratio;

  alpha_ratio /= scaling;
  beta_ratio /= scaling;

  if (lambda.empty()) {
    double lambda_max = LambdaMax(x, y, n_samples, beta_ratio);
    lambda = LogSpace(lambda_max, lambda_max*lambda_min_ratio, n_lambda);
  }

  // The algorithm uses a different penalty construction than
  // glmnet, so convert lambda values to match alpha and beta from scikit-learn.
  for (auto& lambda_val : lambda) {
    // Scaled L2 penalty
    alpha.push_back(alpha_ratio*lambda_val);
    // Scaled L1 penalty
    beta.push_back(beta_ratio*lambda_val);
    lambda_val *= y_scale[0]/scaling;
  }
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
Rcpp::List SetupSgdnet(T&                   x,
                       std::vector<double>& y,
                       const bool           is_sparse,
                       const Rcpp::List&    control) {

  const std::string   family_choice    = Rcpp::as<std::string>(control["family"]);
  const bool          fit_intercept    = Rcpp::as<bool>(control["intercept"]);
  const double        elasticnet_mix   = Rcpp::as<double>(control["elasticnet_mix"]);
  std::vector<double> lambda           = Rcpp::as< std::vector<double> >(control["lambda"]);
  const std::size_t   n_lambda         = Rcpp::as<std::size_t>(control["n_lambda"]);
  const double        lambda_min_ratio = Rcpp::as<double>(control["lambda_min_ratio"]);
  const bool          normalize        = Rcpp::as<bool>(control["normalize"]);
  std::size_t         max_iter         = Rcpp::as<std::size_t>(control["max_iter"]);
  const double        tol              = Rcpp::as<double>(control["tol"]);
  const bool          debug            = Rcpp::as<bool>(control["debug"]);

  std::size_t n_samples  = x.n_rows;
  std::size_t n_features = x.n_cols;

  // Preprocess features
  std::vector<double> x_center;
  std::vector<double> x_scale;
  x_center.reserve(n_features);
  x_scale.reserve(n_features);

  PreprocessFeatures(x,
                     normalize,
                     fit_intercept,
                     x_center,
                     x_scale,
                     is_sparse,
                     n_features,
                     n_samples);

  // Transpose x for more efficient access of samples
  AdaptiveTranspose(x);

  double intercept_decay = is_sparse ? 0.01 : 1.0;

  // Setup family-specific options
  sgdnet::FamilyFactory family_factory;
  std::unique_ptr<sgdnet::Family> family =
    family_factory.NewFamily(family_choice);

  std::size_t n_classes = family->GetNClasses();

  // "Fit" the intercept-only model and compute its deviance = the null deviance
  double null_deviance = family->NullDeviance(y);

  // Preprocess response
  std::vector<double> y_center;
  std::vector<double> y_scale;
  y_center.reserve(n_classes);
  y_scale.reserve(n_classes);

  family->PreprocessResponse(y, y_center, y_scale, fit_intercept);

  // Compute the lambda sequence
  std::vector<double> alpha;
  std::vector<double> beta;

  RegularizationPath(lambda,
                     n_lambda,
                     lambda_min_ratio,
                     elasticnet_mix,
                     x,
                     y,
                     n_samples,
                     alpha,
                     beta,
                     y_scale);

  std::size_t n_penalties = lambda.size();

  // Maximum of sums of squares over samples
  double max_squared_sum = ColNormsMax(x);

  std::vector<double> step_size = family->StepSize(max_squared_sum,
                                                   alpha,
                                                   fit_intercept,
                                                   n_samples);

  // Check if we need the nontrivial prox
  // TODO(jolars): allow more proximal operators
  sgdnet::ProxFactory prox_factory;
  std::string prox_choice = "soft_threshold";
  std::unique_ptr<sgdnet::Prox> prox = prox_factory.NewProx(prox_choice);

  // Setup intercept vector
  std::vector<double> intercept(n_classes, 0.0);
  std::vector< std::vector<double> > intercept_archive(n_penalties);

  // Setup weights matrix and weights archive
  std::vector<double> weights(n_features*n_classes, 0.0);
  std::vector< std::vector<double> > weights_archive;

  // Sum of gradients for weights
  std::vector<double> sum_gradient(n_features*n_classes);

  // Sum of gradients for intercept
  std::vector<double> intercept_sum_gradient(n_classes, 0.0);

  // Gradient memory
  std::vector<double> gradient_memory(n_samples*n_classes);

  // Keep track of the number of as well as which samples are seen
  std::vector<bool> seen(n_samples, false);
  std::size_t n_seen = 0;

  // Keep keep track of successes for each penalty
  std::vector<unsigned int> return_codes;

  // Setup a vector of loss vectors to return upon exit
  std::vector< std::vector<double> > losses_archive;
  std::vector<double> losses;

  // Keep track of number of iteratios per penalty
  std::size_t n_iter = 0;

  // Null deviance on scaled y for computing deviance ratio
  double null_deviance_scaled = family->NullDeviance(y);
  std::vector<double> deviance_ratio;

  // Fit the path of penalty values
  for (std::size_t penalty_ind = 0; penalty_ind < n_penalties; ++penalty_ind) {
    Saga(x,
         y,
         weights,
         fit_intercept,
         intercept,
         intercept_decay,
         intercept_sum_gradient,
         family,
         prox,
         step_size[penalty_ind],
         alpha[penalty_ind],
         beta[penalty_ind],
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
         return_codes,
         losses,
         debug);

    double deviance = Deviance(x,
                               y,
                               weights,
                               intercept,
                               is_sparse,
                               n_samples,
                               n_features,
                               n_classes,
                               family);

    deviance_ratio.push_back(1.0 - deviance/null_deviance_scaled);

    // Rescale and store intercepts and weights for the current solution
    Rescale(weights,
            weights_archive,
            intercept,
            intercept_archive,
            x_center,
            x_scale,
            y_center,
            y_scale,
            n_features,
            n_classes,
            fit_intercept);

    if (debug) {
      // Store losses
      losses_archive.push_back(losses);
      // Reset loss
      losses.clear();
    }
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
Rcpp::List SgdnetCpp(SEXP                 x_in,
                     std::vector<double>& y,
                     const Rcpp::List&    control) {

  // ProfilerStart("/tmp/sgdnet.prof");

  bool is_sparse = Rcpp::as<bool>(control["is_sparse"]);

  if (is_sparse) {
    arma::sp_mat x = Rcpp::as<arma::sp_mat>(x_in);
    return SetupSgdnet(x, y, is_sparse, control);
  } else {
    arma::mat x = Rcpp::as<arma::mat>(x_in);
    return SetupSgdnet(x, y, is_sparse, control);
  }
  // ProfilerStop();
}

