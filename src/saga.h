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

void LaggedUpdate(int k,
                  std::vector<double>& weights,
                  const std::vector<double>& sum_gradient,
                  std::vector<int>& lag,
                  const std::vector<int>& yindices,
                  const std::vector<double>& lag_scaling,
                  const double a) {

  for (const auto& ind : yindices) {
    int lagged_amount = k - lag[ind];
    lag[ind] = k;
    weights[ind] += lag_scaling[lagged_amount]*(a*sum_gradient[ind]);
  }
}

template <typename T>
double SparseDotProduct(const std::vector<double>& weights,
                        int sample_ind,
                        const T& x,
                        const std::vector<int>& yindices) {
  auto inner_product = 0.0;
  auto x_it = x.begin_col(sample_ind);
  for (const auto& ind : yindices) {
    inner_product += (*x_it) * weights[ind];
    ++x_it;
  }

  return inner_product;
}

template <typename T>
void AddWeighted(std::vector<double>& weights,
                 const T& x,
                 int sample_ind,
                 const std::vector<int>& yindices,
                 double a) {
  auto v = 0.0;
  auto x_it = x.begin_col(sample_ind);
  for (const auto& ind : yindices) {
    weights[ind] += a * (*x_it);
  }
}

template <typename T>
void Saga(const T&                                x,
          const std::vector<double>&              y,
          std::vector<double>&                    weights,
          std::unique_ptr<sgdnet::Family>&        family,
          std::unique_ptr<sgdnet::Prox>&          prox,
          const std::vector<double>&              norm_square,
          const double                            step_size,
          const double                            alpha_scaled,
          const double                            beta_scaled,
          std::vector<std::vector<int> >&         nonzero_indices,
          std::vector<double>&                    sum_gradient,
          std::vector<double>&                    gradient_memory,
          const int                               n_samples,
          const int                               n_features,
          const bool                              is_sparse,
          const int                               max_iter,
          const double                            tol,
          int&                                    n_iter,
          std::vector<unsigned int>&              return_codes,
          std::vector<double>&                    losses,
          const bool                              debug) {

  // Are we dealing with a nontrivial prox?
  const bool nontrivial_prox = beta_scaled > 0.0;
  const int prox_ind = static_cast<int>(nontrivial_prox) + 1;

  // Keep track of when each feature was last updated
  std::vector<int> lag(n_features);

  auto reg = beta_scaled;
  auto betak = 1.0;
  auto gamma = step_size;

  double prox_conversion_factor = 1 - (reg*gamma)/(1 + reg*gamma);

  std::vector<double> lag_scaling(n_samples + 2);

  auto geo_sum = 1.0;
  auto mult = prox_conversion_factor;

  lag_scaling[0] = 0.0;
  lag_scaling[1] = 1.0;

  for (auto i = 2; i < n_samples + 2; ++i) {
    geo_sum *= mult;
    lag_scaling[i] = lag_scaling[i - 1] + geo_sum;
  }

  // Store previous weights for computing stopping criteria
  std::vector<double> previous_weights(weights);

  // Keep a vector of the full range of indicies for each row for when
  // we update the full range of weights
  std::vector<int> full_range_indices(n_features);
  std::iota(full_range_indices.begin(), full_range_indices.end(), 0);

  std::vector<int> yindices(full_range_indices);

  int k = 0;

  // Outer loop
  int it_outer = 0;
  for (; it_outer < max_iter; ++it_outer) {

    // Inner loop
    for (auto it_inner = 0; it_inner < n_samples; ++it_inner) {

      int sample_ind =
        it_outer == 0 ? it_inner : std::floor(R::runif(0.0, n_samples));

      // Update the number of samples seen and the seen array
      if (is_sparse)
        yindices = Nonzeros(x.col(sample_ind));

      k++;

      double gamma_prime = gamma*prox_conversion_factor;

      LaggedUpdate(k,
                   weights,
                   sum_gradient,
                   lag,
                   yindices,
                   lag_scaling,
                   -gamma/betak);

      AddWeighted(weights,
                  x,
                  sample_ind,
                  yindices,
                  gradient_memory[sample_ind]*gamma/betak);

      betak *= prox_conversion_factor;

      double activation = betak * SparseDotProduct(weights,
                                                   sample_ind,
                                                   x,
                                                   yindices);
      double new_loc = 0.0;
      double cnew = 0.0;

      family->Prox(activation,
                   y[sample_ind],
                   gamma_prime*norm_square[sample_ind],
                   new_loc,
                   cnew);

      double cold = gradient_memory[sample_ind];
      double cchange = cnew - cold;
      gradient_memory[sample_ind] = cnew;

      double sg = family->Gradient(new_loc, y[sample_ind]);

      AddWeighted(weights, x, sample_ind, yindices, -cnew*gamma_prime/betak);
      AddWeighted(sum_gradient, x, sample_ind, yindices, cchange/n_samples);

    } // inner loop

    double gscaling = -gamma/betak;

    // unlag the vector

    for (auto ind = 0; ind < n_features; ++ind) {
      int lagged_amount = k - lag[ind];
      if (lagged_amount > 0) {
        lag[ind] = k;
        weights[ind] +=
          (lag_scaling[lagged_amount + 1] - 1)*gscaling*sum_gradient[ind];
      }
      weights[ind] = betak*weights[ind];
    }

    betak = 1.0;

    // compute loss for the current solution if debugging
    // if (debug)
    //   EpochLoss(x,
    //             y,
    //             weights,
    //             intercept,
    //             family,
    //             alpha_scaled,
    //             beta_scaled,
    //             n_samples,
    //             n_classes,
    //             is_sparse,
    //             losses);

    // check termination conditions
    auto max_change = 0.0;
    auto max_weight = 0.0;

    for (auto i = 0; i < weights.size(); ++i) {
      auto abs_weight = std::abs(weights[i]);
      if (abs_weight > max_weight)
        max_weight = abs_weight;

      auto change = std::abs(weights[i] - previous_weights[i]);
      if (change > max_change)
        max_change = change;
      previous_weights[i] = weights[i];
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
