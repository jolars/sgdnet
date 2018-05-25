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

#include <RcppArmadillo.h>
#include "utils.h"
#include "math.h"
#include "families.h"
#include "constants.h"
#include "prox.h"

//' Perform Lagged Updates
//'
//' @param weights the weights matrix
//' @param wscale the weights scale
//' @param current_nonzero_indices a vector of indices for the nonzero elements
//'   of the current data point
//' @param n_samples the number of data points
//' @param n_classes the number of classes for the outcome
//' @param cumulative_sums storage for cumulative sums
//' @param feature_history keeps track of the iteration at which each
//'   feature was last updated
//' @param nontrivial_prox nontrivial proximal operator?
//' @param sum_gradient gradient sum storage
//' @param reset TRUE if wscale is to be reset and weights rescaled
//' @param it_inner the current iteration in the inner loop
//'
//' @return Weights, cumulative_sums, and feature_history are updated.
//'
//' @noRd
//' @keywords internal
void LaggedUpdate(arma::mat&                     weights,
                  double                         wscale,
                  const arma::uvec&              nonzero_indices,
                  arma::uword                    n_samples,
                  arma::uword                    n_classes,
                  arma::mat&                     cumulative_sums,
                  arma::uvec&                    feature_history,
                  bool                           nontrivial_prox,
                  const arma::mat&               sum_gradient,
                  bool                           reset,
                  arma::uword                    it_inner,
                  std::unique_ptr<sgdnet::Prox>& prox) {

  arma::uvec::const_iterator feature_itr = nonzero_indices.begin();
  arma::uvec::const_iterator feature_end = nonzero_indices.end();

  for (; feature_itr != feature_end; ++feature_itr) {

    arma::uword feature_ind = (*feature_itr);

    arma::rowvec cum_sum = cumulative_sums.row(it_inner - 1);

    if (feature_history(feature_ind) != 0) {
      cum_sum -= cumulative_sums.row(feature_history(feature_ind) - 1);
    }

    if (nontrivial_prox) {

      for (arma::uword class_ind = 0; class_ind < n_classes; ++class_ind) {

        if (std::abs(sum_gradient(feature_ind, class_ind)*cum_sum(0))
              < cum_sum(1)) {

          weights(feature_ind, class_ind) -=
          cum_sum(0)*sum_gradient(feature_ind, class_ind);
          weights(feature_ind, class_ind) =
            prox->Evaluate(weights(feature_ind, class_ind), cum_sum(1));

        } else {

          arma::sword last_update_ind = feature_history(feature_ind) - 1;

          if (last_update_ind == -1)
            last_update_ind = it_inner - 1;

          for (arma::uword lagged_ind = it_inner - 1;
               lagged_ind > last_update_ind - 1;
               --lagged_ind) {

            // Grad and prox steps
            arma::rowvec steps(cumulative_sums.n_cols);

            if (lagged_ind > 0)
              steps = cumulative_sums.row(lagged_ind)
              - cumulative_sums.row(lagged_ind - 1);
            else
              steps = cumulative_sums.row(lagged_ind);

            weights(feature_ind, class_ind) -=
              sum_gradient(feature_ind, class_ind)*steps(0);
            weights(feature_ind, class_ind) =
              prox->Evaluate(weights(feature_ind, class_ind), cum_sum(1));
          }
        }
      }
    } else { // Trivial prox
      weights.row(feature_ind) -= cum_sum(0)*sum_gradient.row(feature_ind);
    }
    if (!reset) {
      feature_history(feature_ind) = it_inner;
    }
  } // for each feature

  if (reset) {
    weights *= wscale;

    if (!(weights.is_finite()))
      Rcpp::stop("non-finite weights.");

    feature_history.fill(it_inner % n_samples);
    cumulative_sums.row(it_inner - 1).zeros();
  }
}

//' Predict Sample
//'
//' @param x sample
//' @param weights weights
//' @param wscale scale for weights
//' @param intercept intercept
//'
//' @return The prediction at the current sample
//'
//' @noRd
//' @keywords internal
template <typename T>
arma::rowvec PredictSample(const T&            x,
                           const arma::mat&    weights,
                           const double        wscale,
                           const arma::rowvec& intercept) {

  return x.t()*(wscale*weights) + intercept;
}

//' Update the intercept
//'
//' @param gradient gradient of current sample
//' @param gradient_memory memory of previously computed gradients
//' @param intercept_sum_gradient sum of intercept gradients
//' @param intercept_correction the correction to be applied to the gradient
//' @param intercept the intercept
//' @param intercept_decay modifier to shrink the learning rate for the
//'   intercept
//'
//' @return `intercept` and `intercept_sum_gradient` are updated
//' @noRd
//' @keywords internal
void UpdateIntercept(const arma::rowvec& gradient,
                     const arma::rowvec& gradient_memory,
                     arma::rowvec&       intercept_sum_gradient,
                     arma::rowvec&       intercept_correction,
                     arma::rowvec&       intercept,
                     const double        intercept_decay,
                     const double        step_size,
                     const arma::uword   n_seen) {
  intercept_correction = gradient - gradient;
  intercept_sum_gradient += intercept_correction;
  intercept_correction *= step_size*(1.0 - 1.0/n_seen);
  intercept -= step_size*intercept_sum_gradient/n_seen*intercept_decay
    + intercept_correction;
  if (!intercept.is_finite())
    Rcpp::stop("non-finite intercepts.");
}

//' SAGA algorithm
//'
//' @param x feature matrix
//' @param y response matrix
//' @param family response type
//' @param fit_intercept whether the intercept should be fit
//' @param intercept_decay intercept updates are scaled by
//'   this decay factor to avoid intercept oscillation when features are
//'   sparse
//' @param alpha l2-regularization penalty
//' @param beta l1-regularization penalty
//' @param normalize whether to normalize x
//' @param max_iter maximum number of iterations
//' @param debug if `TRUE`, we are debugging and should return loss
//' @param is_sparse is x sparse?
//'
//' @return Updates weights, intercept, sum_gradient, intercept_sum_gradient,
//'   gradient_memory.
//'
//' @noRd
//' @keywords internal, programming
template <typename T>
void Saga(const T&                          x,
          const arma::mat&                  y,
          arma::mat&                        weights,
          const bool                        fit_intercept,
          arma::rowvec&                     intercept,
          const double                      intercept_decay,
          arma::rowvec&                     intercept_sum_gradient,
          std::unique_ptr<sgdnet::Family>&  family,
          std::unique_ptr<sgdnet::Prox>&    prox,
          const double                      step_size,
          const double                      alpha_scaled,
          const double                      beta_scaled,
          arma::mat&                        sum_gradient,
          arma::mat&                        gradient_memory,
          arma::uvec&                       seen,
          arma::uword&                      n_seen,
          const arma::uword                 n_samples,
          const arma::uword                 n_features,
          const arma::uword                 n_classes,
          const bool                        is_sparse,
          const arma::uword                 max_iter,
          const double                      tol,
          arma::uword&                      n_iter,
          arma::uword&                      return_code,
          std::vector<double>&              losses,
          const bool                        debug) {

  // Are we dealing with a nontrivial prox?
  bool nontrivial_prox = beta_scaled > 0.0;

  // Keep track of when each feature was last updated
  arma::uvec feature_history(n_features, arma::fill::zeros);

  // Gradient correction matrix
  arma::mat gradient_correction(arma::size(weights));

  // Intercept correction vector
  arma::rowvec intercept_correction;
  if (fit_intercept)
    intercept_correction.set_size(n_classes);

  // Store previous weights for computing stopping criteria
  arma::mat previous_weights(weights);

  // Scale of weights
  double wscale = 1.0;

  // Store a matrix of cumulative sums, prox sums in second column
  arma::mat cumulative_sums(n_samples, 1 + nontrivial_prox, arma::fill::zeros);

  // Precomputated stepsize
  double wscale_update = 1.0 - step_size*alpha_scaled;

  // Keep a vector of the full range of indicies for each row for when
  // we update the full range of weights
  arma::uvec full_range_indices = arma::regspace<arma::uvec>(0, n_features - 1);
  arma::uvec nonzero_indices;
  if (!is_sparse)
    nonzero_indices = full_range_indices;

  // Scalars for computing stopping criteria
  double max_change = 0.0;
  double max_weight = 0.0;

  // Vector to store prediction
  arma::rowvec prediction(n_classes);

  // Vector to store gradient
  arma::rowvec gradient(n_classes);

  // Outer loop
  arma::uword it_outer = 0;
  for (; it_outer < max_iter; ++it_outer) {

    // Inner loop
    for (arma::uword it_inner = 0; it_inner < n_samples; ++it_inner) {

      // Extract a random sample
      arma::uword sample_ind = std::floor(R::runif(0.0, n_samples));

      // Update the number of samples seen and the seen array
      if (!seen(sample_ind)) {
        n_seen++;
        seen(sample_ind) = true;
      }

      if (is_sparse)
        nonzero_indices = Nonzeros(x.col(sample_ind));

      if (it_inner > 0)
        LaggedUpdate(weights,
                     wscale,
                     nonzero_indices,
                     n_samples,
                     n_classes,
                     cumulative_sums,
                     feature_history,
                     nontrivial_prox,
                     sum_gradient,
                     false,
                     it_inner,
                     prox);

      prediction = PredictSample(x.col(sample_ind),
                                 weights,
                                 wscale,
                                 intercept);

      gradient = family->Gradient(prediction, y.row(sample_ind));

      // L2-regularization by rescaling the weights
      wscale *= wscale_update;

      // Update the sum of gradients
      gradient_correction =
        x.col(sample_ind)*(gradient - gradient_memory.row(sample_ind));

      weights -= (gradient_correction*step_size*(1.0 - 1.0/n_seen)/wscale);
      sum_gradient += gradient_correction;

      if (fit_intercept) {
        UpdateIntercept(gradient,
                        gradient_memory.row(sample_ind),
                        intercept_sum_gradient,
                        intercept_correction,
                        intercept,
                        intercept_decay,
                        step_size,
                        n_seen);
      }

      // Update the gradient memory for this sample
      gradient_memory.row(sample_ind) = gradient;

      // Update cumulative sums
      if (it_inner == 0) {
        cumulative_sums(0, 0) = step_size/(wscale*n_seen);
        if (nontrivial_prox)
          cumulative_sums(0, 1) = step_size*beta_scaled/wscale;
      } else {
        cumulative_sums(it_inner, 0) =
          cumulative_sums(it_inner - 1, 0) + (step_size/(wscale*n_seen));
        if (nontrivial_prox)
          cumulative_sums(it_inner, 1) =
            cumulative_sums(it_inner - 1, 1) + (step_size*beta_scaled/wscale);
      }

      // if wscale is too small, reset the scale
      if (wscale < sgdnet::SMALL) {
        LaggedUpdate(weights,
                     wscale,
                     full_range_indices,
                     n_samples,
                     n_classes,
                     cumulative_sums,
                     feature_history,
                     nontrivial_prox,
                     sum_gradient,
                     true,
                     it_inner + 1,
                     prox);
        wscale = 1.0;
      }
    } // inner loop

    // scale the weights for every epoch and reset the JIT update system
    LaggedUpdate(weights,
                 wscale,
                 full_range_indices,
                 n_samples,
                 n_classes,
                 cumulative_sums,
                 feature_history,
                 nontrivial_prox,
                 sum_gradient,
                 true,
                 n_samples,
                 prox);

    wscale = 1.0;

    // compute loss for the current solution if debugging
    if (debug) {
      arma::mat pred = x.t()*weights + arma::repmat(intercept, n_samples, 1);
      double loss = family->Loss(pred, y)
        + alpha_scaled*std::pow(arma::norm(weights), 2);
      losses.push_back(loss);
    }

    // check termination conditions
    max_weight = arma::abs(weights).max();
    max_change = arma::abs(weights - previous_weights).max();
    previous_weights = weights;

    if ((max_weight != 0.0 && max_change/max_weight <= tol)
          || (max_weight == 0.0 && max_change == 0.0)) {
      break;
    }
  } // outer loop
  it_outer++;

  // Update accumulated number of epochs
  n_iter += it_outer;

  if (it_outer == max_iter) {
    // Iteration limit reached
    return_code = 1;
  } else {
    // Successful convergence
    return_code = 0;
  }
}
