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
//' @param nonzero_ptr a pointer to a vector of indices for the nonzero elements
//'   of the current sample
//' @param n_samples the number of data points
//' @param n_classes the number of classes for the outcome
//' @param cumulative_sums storage for cumulative sums
//' @param cumulative_sums_prox storage for cumulative sums for the proximal
//'   operator
//' @param feature_history keeps track of the iteration at which each
//'   feature was last updated
//' @param nontrivial_prox nontrivial proximal operator?
//' @param sum_gradient gradient sum storage
//' @param reset TRUE if wscale is to be reset and weights rescaled
//' @param it_inner the current iteration in the inner loop
//' @param prox a proximal operator
//'
//' @return Weights, cumulative_sums, and feature_history are updated.
//'
//' @noRd
void LaggedUpdate(std::vector<double>&            weights,
                  double                          wscale,
                  std::vector<std::size_t>       *nonzero_ptr,
                  const std::size_t               n_samples,
                  const std::size_t               n_classes,
                  std::vector<double>&            cumulative_sums,
                  std::vector<double>&            cumulative_sums_prox,
                  std::vector<std::size_t>&       feature_history,
                  const bool                      nontrivial_prox,
                  const std::vector<double>&      sum_gradient,
                  const bool                      reset,
                  const std::size_t               it_inner,
                  std::unique_ptr<sgdnet::Prox>&  prox) {

  for (auto&& feature_ind : *nonzero_ptr) {

    double cum_sum = cumulative_sums[it_inner - 1];

    int last_update_ind = feature_history[feature_ind] - 1;

    if (feature_history[feature_ind] != 0)
      cum_sum -= cumulative_sums[last_update_ind];

    if (nontrivial_prox) {
      double cum_sum_prox = cumulative_sums_prox[it_inner - 1];

      if (feature_history[feature_ind] != 0)
        cum_sum_prox -= cumulative_sums_prox[last_update_ind];

      for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind) {
        std::size_t f_idx = feature_ind*n_classes + class_ind;

        if (std::abs(sum_gradient[f_idx]*cum_sum) < cum_sum_prox) {

          //weights[f_idx] -= cum_sum*sum_gradient[f_idx];
          weights[f_idx] =
            prox->Evaluate(weights[f_idx] - cum_sum*sum_gradient[f_idx],
                           cum_sum_prox);

        } else {

          if (last_update_ind == -1)
            last_update_ind = it_inner - 1;

          for (std::size_t lagged_ind = it_inner - 1;
               lagged_ind > last_update_ind - 1;
               --lagged_ind) {

            // Grad and prox steps
            double grad_step;
            double prox_step;

            if (lagged_ind > 0) {
              grad_step = cumulative_sums[lagged_ind]
                          - cumulative_sums[lagged_ind - 1];
              prox_step = cumulative_sums_prox[lagged_ind]
                          - cumulative_sums_prox[lagged_ind - 1];
            } else {
              grad_step = cumulative_sums[lagged_ind];
              prox_step = cumulative_sums_prox[lagged_ind];
            }

            weights[f_idx] =
              prox->Evaluate(weights[f_idx] - sum_gradient[f_idx]*grad_step,
                             prox_step);
          }
        }
        if (reset) {
          weights[f_idx] *= wscale;
          if (!std::isfinite(weights[f_idx]))
            Rcpp::stop("non-finite weights.");
        }
      } // for class_ind (nontrivial prox)
    } else { // Trivial prox
      for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind) {
        std::size_t f_idx = feature_ind*n_classes + class_ind;
        weights[f_idx] -= cum_sum*sum_gradient[f_idx];
        if (reset)
          weights[f_idx] *= wscale;
      }
    }

    feature_history[feature_ind] = reset ? it_inner % n_samples : it_inner;
  } // for each feature

  if (reset) {
    cumulative_sums[it_inner - 1] = 0.0;
    if (nontrivial_prox)
      cumulative_sums_prox[it_inner - 1] = 0.0;
  }
}

//' Update the gradient
//'
//' Update the gradient and store it in `gradient_memory` and update
//' coefficients
//'
//' @param x the current sample
//' @param nonzero_ptr vector of indices of nonzero elements in the
//'   current sample
//' @param weights coefficients
//' @param gradient gradient for current sample
//' @param gradient_memory memory of gradients for each sample
//' @param sum_gradient gradient sum
//' @param step_size step size
//' @param wscale scale for weights
//' @param n_seen number of samples seen so far
//' @param n_classes pseudo-number of classes
//' @param sample_ind index of current sample
//'
//' @return Updates weights and sum_gradient.
//'
//' @noRd
template <typename T>
void UpdateWeights(const T&                        x,
                   std::vector<std::size_t>       *nonzero_ptr,
                   std::vector<double>&            weights,
                   std::vector<double>&            intercept,
                   const std::vector<double>&      gradient,
                   const std::vector<double>&      gradient_memory,
                   std::vector<double>&            sum_gradient,
                   std::vector<double>&            intercept_sum_gradient,
                   const double                    step_size,
                   const double                    wscale,
                   const double                    intercept_decay,
                   const std::size_t               n_seen,
                   const std::size_t               n_classes,
                   const std::size_t               sample_ind,
                   const bool                      fit_intercept) {
  auto x_itr = x.begin_col(sample_ind);

  for (auto&& feature_ind : *nonzero_ptr) {
    for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind) {
      std::size_t s_idx = sample_ind*n_classes + class_ind;
      std::size_t f_idx = feature_ind*n_classes + class_ind;

      double gradient_correction =
        (*x_itr)*(gradient[class_ind] - gradient_memory[s_idx]);
      weights[f_idx] -= gradient_correction*step_size*(1.0 - 1.0/n_seen)/wscale;
      sum_gradient[f_idx] += gradient_correction;

      if (fit_intercept) {
        gradient_correction = gradient[class_ind] - gradient_memory[s_idx];
        intercept_sum_gradient[class_ind] += gradient_correction;
        intercept[class_ind] -=
          step_size*intercept_sum_gradient[class_ind]/n_seen*intercept_decay
          + gradient_correction*step_size*(1.0 - 1.0/n_seen);
      }
    }
    ++x_itr;
  }
}

//' SAGA algorithm
//'
//' @param x feature matrix
//' @param y response matrix
//' @param weights coefficients
//' @param fit_intercept whether to fit the intercept
//' @param intercept intercept
//' @param intercept_decay weight of intercept update, which is different
//'   for the sparse implementation
//' @param intercept_sum_gradient gradient sum for the intercept
//' @param family response type
//' @param prox proximal operator
//' @param step_size step size
//' @param alpha_scaled scaled l2-penalty weight
//' @param beta_scaled scaled l1-penalty weight
//' @param sum_gradient gradient sum for the weights
//' @param gradient_memory storage for gradients for each sample
//' @param seen vector of indices for whether the sample has been seen
//'   previously
//' @param n_seen number of previously seen samples
//' @param n_samples number of samples
//' @param n_features number of features (variables)
//' @param n_classes pseudo-number of classes
//' @param is_sparse whether x is sparse
//' @param max_iter maximum number of iterations
//' @param tol treshold for convergence (stops if max weight/max change
//'   in weights < tol)
//' @param n_iter number of accumulated effective passes
//' @param return_codes vector of return codes for each fit
//' @param losses vector of losses for each fit and pass
//' @param debug whether diagnostic information should be computed
//'
//' @return Updates weights, intercept, sum_gradient, intercept_sum_gradient,
//'   gradient_memory.
//'
//' @noRd
template <typename T>
void Saga(const T&                                x,
          const std::vector<double>&              y,
          std::vector<double>&                    weights,
          const bool                              fit_intercept,
          std::vector<double>&                    intercept,
          const double                            intercept_decay,
          std::vector<double>&                    intercept_sum_gradient,
          std::unique_ptr<sgdnet::Family>&        family,
          std::unique_ptr<sgdnet::Prox>&          prox,
          const double                            step_size,
          const double                            alpha_scaled,
          const double                            beta_scaled,
          std::vector<std::vector<std::size_t> >& nonzero_indices,
          std::vector<double>&                    sum_gradient,
          std::vector<double>&                    gradient_memory,
          std::vector<bool>&                      seen,
          std::size_t&                            n_seen,
          const std::size_t                       n_samples,
          const std::size_t                       n_features,
          const std::size_t                       n_classes,
          const bool                              is_sparse,
          const std::size_t                       max_iter,
          const double                            tol,
          std::size_t&                            n_iter,
          std::vector<unsigned int>&              return_codes,
          std::vector<double>&                    losses,
          const bool                              debug) {

  // Are we dealing with a nontrivial prox?
  bool nontrivial_prox = beta_scaled > 0.0;

  // Keep track of when each feature was last updated
  std::vector<std::size_t> feature_history(n_features, 0);

  // Store previous weights for computing stopping criteria
  std::vector<double> previous_weights(weights);

  // Scale of weights
  double wscale = 1.0;

  // Store a matrix of cumulative sums, prox sums in second column
  std::vector<double> cumulative_sums(n_samples);
  std::vector<double> cumulative_sums_prox(n_samples);

  // Precomputated stepsize
  double wscale_update = 1.0 - step_size*alpha_scaled;

  // Keep a vector of the full range of indicies for each row for when
  // we update the full range of weights
  std::vector<std::size_t> full_range_indices(n_features);
  std::iota(full_range_indices.begin(), full_range_indices.end(), 0);

  std::vector<std::size_t> *nonzero_ptr;

  if (!is_sparse)
    nonzero_ptr = &full_range_indices;

  // Vector to store prediction
  std::vector<double> prediction(n_classes);
  prediction.reserve(n_classes);

  // Vector to store gradient
  std::vector<double> gradient;
  gradient.reserve(n_classes);

  // Outer loop
  std::size_t it_outer = 0;
  for (; it_outer < max_iter; ++it_outer) {

    // Inner loop
    for (std::size_t it_inner = 0; it_inner < n_samples; ++it_inner) {

      // Extract a random sample
      std::size_t sample_ind = std::floor(R::runif(0.0, n_samples));

      // Update the number of samples seen and the seen array
      if (!seen[sample_ind]) {
        n_seen++;
        seen[sample_ind] = true;
        if (is_sparse)
          nonzero_indices[sample_ind] = Nonzeros(x.col(sample_ind));
      }

      if (is_sparse)
        nonzero_ptr = &nonzero_indices[sample_ind];

      if (it_inner > 0)
        LaggedUpdate(weights,
                     wscale,
                     nonzero_ptr,
                     n_samples,
                     n_classes,
                     cumulative_sums,
                     cumulative_sums_prox,
                     feature_history,
                     nontrivial_prox,
                     sum_gradient,
                     false,
                     it_inner,
                     prox);

      PredictSample(prediction,
                    x,
                    nonzero_ptr,
                    weights,
                    wscale,
                    intercept,
                    n_classes,
                    sample_ind);

      for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind) {
        gradient[class_ind] =
          family->Gradient(prediction[class_ind],
                           y[sample_ind*n_classes + class_ind]);
      }

      // L2-regularization by rescaling the weights
      wscale *= wscale_update;

      UpdateWeights(x,
                    nonzero_ptr,
                    weights,
                    intercept,
                    gradient,
                    gradient_memory,
                    sum_gradient,
                    intercept_sum_gradient,
                    step_size,
                    wscale,
                    intercept_decay,
                    n_seen,
                    n_classes,
                    sample_ind,
                    fit_intercept);

      // Update the gradient memory for this sample
      for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind)
        gradient_memory[sample_ind*n_classes + class_ind] = gradient[class_ind];

      // Update cumulative sums
      if (it_inner == 0) {
        cumulative_sums[0] = step_size/(wscale*n_seen);
        if (nontrivial_prox)
          cumulative_sums_prox[0] = step_size*beta_scaled/wscale;
      } else {
        cumulative_sums[it_inner] =
          cumulative_sums[it_inner - 1] + step_size/(wscale*n_seen);
        if (nontrivial_prox)
          cumulative_sums_prox[it_inner] =
            cumulative_sums_prox[it_inner - 1] + step_size*beta_scaled/wscale;
      }

      // if wscale is too small, reset the scale
      if (wscale < sgdnet::SMALL) {
        LaggedUpdate(weights,
                     wscale,
                     &full_range_indices,
                     n_samples,
                     n_classes,
                     cumulative_sums,
                     cumulative_sums_prox,
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
                 &full_range_indices,
                 n_samples,
                 n_classes,
                 cumulative_sums,
                 cumulative_sums_prox,
                 feature_history,
                 nontrivial_prox,
                 sum_gradient,
                 true,
                 n_samples,
                 prox);

    wscale = 1.0;

    // compute loss for the current solution if debugging
    if (debug)
      EpochLoss(x,
                y,
                weights,
                intercept,
                family,
                alpha_scaled,
                beta_scaled,
                n_samples,
                n_classes,
                is_sparse,
                losses);

    // check termination conditions
    double max_weight = 0.0;
    double max_change = 0.0;

    auto previous_weight_itr = previous_weights.begin();

    for (const auto& weight : weights) {
      max_weight = std::max(max_weight, std::abs(weight));
      max_change = std::max(max_change,
                            std::abs(weight - (*previous_weight_itr)));
      *previous_weight_itr = weight;
      ++previous_weight_itr;
    }

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
    return_codes.push_back(1);
  } else {
    // Successful convergence
    return_codes.push_back(0);
  }
}
