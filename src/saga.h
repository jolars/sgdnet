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
#include "math.h"
#include "families.h"
#include "penalties.h"
#include "prox.h"
#include "constants.h"
#include "prox.h"

//' Lagged updates
//'
//' @param k current iteration
//' @param w weights vector
//' @param n_features number of features
//' @param g_sum gradient sum
//' @param lag iteration at which the features were last updated
//' @param x the feature matrix. Sparse or dense Eigen object.
//' @param s_ind the index of the current sample
//' @param lag_scaling geometric sum for lagged updates
//' @param penalty object of Penalty class
//'
//' @return Updates weights and lag.
template <typename Penalty>
inline void LaggedUpdate(const unsigned             k,
                         Eigen::MatrixXd&           w,
                         const unsigned             n_features,
                         const Eigen::MatrixXd&     g_sum,
                         std::vector<unsigned>&     lag,
                         const Eigen::MatrixXd&     x,
                         const unsigned             s_ind,
                         const std::vector<double>& lag_scaling,
                         const double               wscale,
                         const Penalty&             penalty) noexcept {

  for (unsigned j = 0; j < n_features; ++j) {

    auto lagged_amount = k - lag[j];

    if (lagged_amount != 0) {
      penalty(w, j, wscale, lag_scaling[lagged_amount], g_sum);
      lag[j] = k;
    }
  }
}

template <typename Penalty>
inline void LaggedUpdate(const unsigned                     k,
                         Eigen::MatrixXd&                   w,
                         const unsigned                     n_features,
                         const Eigen::MatrixXd&             g_sum,
                         std::vector<unsigned>&             lag,
                         const Eigen::SparseMatrix<double>& x,
                         const unsigned                     s_ind,
                         const std::vector<double>&         lag_scaling,
                         const double                       wscale,
                         const Penalty&                     penalty) noexcept {

  for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it) {

    auto j = it.index();
    auto lagged_amount = k - lag[j];

    if (lagged_amount != 0) {
      penalty(w, j, wscale, lag_scaling[lagged_amount], g_sum);
      lag[j] = k;
    }
  }
}

//' Weighted addition
//'
//' Updates `y` with a weighted sample in `x`
//'
//' @param a weights or gradient vector
//' @param x the feature matrix. Sparse or dense Eigen object.
//' @param scaling step size
//'
//' @return Updates `y` with `x` scaled.
inline void AddWeighted(Eigen::MatrixXd&           a,
                        const Eigen::MatrixXd&     x,
                        const unsigned             s_ind,
                        const unsigned             n_features,
                        const unsigned             n_classes,
                        const std::vector<double>& g_change,
                        const double               scaling) noexcept {

  for (unsigned f_ind = 0; f_ind < n_features; ++f_ind)
    for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind)
      a(c_ind, f_ind) += g_change[c_ind] * scaling * x(f_ind, s_ind);
}

inline void AddWeighted(Eigen::MatrixXd&                   a,
                        const Eigen::SparseMatrix<double>& x,
                        const unsigned                     s_ind,
                        const unsigned                     n_features,
                        const unsigned                     n_classes,
                        const std::vector<double>&         g_change,
                        const double                       scaling) noexcept {

  for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it)
    for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind)
      a(c_ind, it.index()) += g_change[c_ind] * scaling * it.value();
}

//' Reset weights and lag
//'
//' Scales and resets the weights and lag
//'
//' @param k current iteration
//' @param w weights vector
//' @param g_sum gradient sum
//' @param lag_scaling geometric sum for lagged updates
//' @param lag iteration at which the features were last updated
//' @param n_features number of features
//' @param prox_scaling step size for the projection step
//' @param grad_scaling step size for gradient step
//' @param nontrivial_prox true if non-trivial (not L2) update
//' @param prox pointer to the proximal operator
//'
//' @return Unlags the coefficients by adding the lagged updates.
template <typename Penalty>
inline
double
Reset(const unsigned         k,
      Eigen::MatrixXd&       w,
      const Eigen::MatrixXd& g_sum,
      std::vector<double>&   lag_scaling,
      std::vector<unsigned>& lag,
      const unsigned         n_features,
      double                 wscale,
      const Penalty&         penalty) noexcept {

  for (unsigned j = 0; j < n_features; ++j) {

    auto lagged_amount = k - lag[j];

    if (lagged_amount != 0)
      penalty(w, j, wscale, lag_scaling[lagged_amount], g_sum);
  }

  w.array() *= wscale;

  return 1.0;
}

//' The SAGA algorithm
//'
//' @param x the feature matrix
//' @param y response vector or vectorized response matrix
//' @param fit_intercept whether the intercept should be fit
//' @param intercept_decay adjustment of learning rate for intercept,
//'   which is lower for sparse features to guard against intercept
//'   oscillation
//' @param intercept the vector. Initialized to zero but will be stored
//'   and continually updated along the regularization path to support
//'   warm starts
//' @param w weights. Updated in the same manner as `intercept`.
//' @param family a pointer to the Family object
//' @param prox a pointer to the Prox object
//' @param gamma step size
//' @param alpha L2-regularization penalty strength
//' @param beta L1-regularization penalty strength
//' @param g_sum gradient sum
//' @param g_sum_intercept gradient sum for intercept
//' @param g gradient memory
//' @param n_samples number of samples
//' @param n_features number of features
//' @param n_classes number of classes
//' @param max_iter maximum number of iterations allowed
//' @param tol tolerance threshold for stopping criteria
//' @param n_iter accumulated number of epochs (outer iterations) along
//'   the path
//' @param return_codes a vector storage for return codes from each
//'   fit along the regularization path. 0 means the algorithm converged. 1
//'   means that `max_iter` was reached before convergence
//' @param losses a temporary storage for loss from vectors. Only used if
//'   `debug` is true.
//' @param debug whether we are debuggin and should store loss from the
//'   fit inside `losses`.
//'
//' @return Updates `w`, `intercept`, `g_sum`, `g_sum_intercept`, `g`,
//'   `n_iter`, `return_codes`, and possibly `losses`.
template <typename T, typename Family, typename Penalty>
void Saga(const T&               x,
          const Eigen::MatrixXd& y,
          std::vector<double>&   intercept,
          const bool             fit_intercept,
          const double           intercept_decay,
          Eigen::MatrixXd&       w,
          const Family&          family,
          Penalty&               penalty,
          const double           gamma,
          const double           alpha,
          const double           beta,
          Eigen::MatrixXd&       g_memory,
          Eigen::MatrixXd&       g_sum,
          std::vector<double>&   g_sum_intercept,
          const unsigned         n_samples,
          const unsigned         n_features,
          const unsigned         n_classes,
          const unsigned         max_iter,
          const double           tol,
          unsigned&              n_iter,
          std::vector<unsigned>& return_codes,
          std::vector<double>&   losses,
          const bool             debug) noexcept {

  using namespace std;

  // Are we dealing with a nontrivial prox?
  const bool nontrivial_prox = beta > 0.0;

  // Keep track of when each feature was last updated
  std::vector<unsigned> lag(n_features);

  double wscale = 1.0;

  vector<double> lag_scaling;
  lag_scaling.reserve(n_samples + 1);
  lag_scaling.push_back(0.0);
  lag_scaling.push_back(1.0);
  double geo_sum = 1.0;
  double wscale_update = 1.0 - alpha*gamma;

  for (unsigned i = 2; i < n_samples + 1; ++i) {
    geo_sum *= wscale_update;
    double tmp = lag_scaling.back() + geo_sum;
    lag_scaling.push_back(tmp);
  }

  penalty.setParameters(gamma, alpha, beta);

  // Setup gradient vectors
  vector<double> g(n_classes);
  vector<double> g_change(n_classes);

  // Vector for storing current predictions
  vector<double> prediction(n_classes);

  // Store previous weights for computing stopping criteria
  Eigen::MatrixXd w_previous(w);

  // Outer loop
  unsigned it_outer = 0;
  bool converged = false;
  do {
    // Inner loop
    for (unsigned it_inner = 0; it_inner < n_samples; ++it_inner) {

      // Pull a sample
      unsigned s_ind = floor(R::runif(0.0, n_samples));

      LaggedUpdate(it_inner,
                   w,
                   n_features,
                   g_sum,
                   lag,
                   x,
                   s_ind,
                   lag_scaling,
                   wscale,
                   penalty);

      PredictSample(prediction,
                    w,
                    wscale,
                    n_features,
                    n_classes,
                    s_ind,
                    x,
                    intercept);

      family.Gradient(prediction, y, s_ind, g);

      for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind) {
        g_change[c_ind] = g[c_ind] - g_memory(c_ind, s_ind);
        // Store current gradient
        g_memory(c_ind, s_ind) = g[c_ind];
      }

      // Rescale and unlag weights whenever wscale becomes too small
      if (wscale < sgdnet::SMALL) {
        wscale = Reset(it_inner,
                       w,
                       g_sum,
                       lag_scaling,
                       lag,
                       n_features,
                       wscale,
                       penalty);
        lag.assign(lag.size(), it_inner);
      }

      wscale *= wscale_update;

      // Update coefficients (w) with sparse step (with L2 scaling)
      AddWeighted(w,
                  x,
                  s_ind,
                  n_features,
                  n_classes,
                  g_change,
                  -gamma/wscale);

      if (fit_intercept) {
        for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind) {
          g_sum_intercept[c_ind] += g_change[c_ind]/n_samples;
          intercept[c_ind] -= gamma*g_sum_intercept[c_ind]*intercept_decay
                              + g_change[c_ind]/n_samples;
        }
      }

      // Gradient-average step
      LaggedUpdate(it_inner + 1,
                   w,
                   n_features,
                   g_sum,
                   lag,
                   x,
                   s_ind,
                   lag_scaling,
                   wscale,
                   penalty);

      // Update the gradient average
      AddWeighted(g_sum,
                  x,
                  s_ind,
                  n_features,
                  n_classes,
                  g_change,
                  1.0/n_samples);

    } // Outer loop

    // Unlag and rescale coefficients
    wscale = Reset(n_samples,
                   w,
                   g_sum,
                   lag_scaling,
                   lag,
                   n_features,
                   wscale,
                   penalty);
    lag.assign(lag.size(), 0);

    if (debug) {
      double loss = EpochLoss(x,
                              y,
                              w,
                              intercept,
                              family,
                              alpha,
                              beta,
                              n_samples,
                              n_features,
                              n_classes);
      losses.push_back(loss);
    }

    converged = CheckConvergence(w, w_previous, tol);
    ++it_outer;

  } while (!converged && it_outer < max_iter); // outer loop

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

