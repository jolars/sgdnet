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
// ARE DISCLAIMED. IN NO EVENT SHALL THE alphaENTS OR CONTRIBUTORS BE LIABLE FOR
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
#include "constants.h"
#include "prox.h"

//' Lagged updates for L2-regularized regression
//'
//' @param k current iteration
//' @param w weights vector
//' @param n_features number of features
//' @param g_sum gradient sum
//' @param lag iteration at which the features were last updated
//' @param x the feature matrix. Sparse or dense Eigen object.
//' @param s_ind the index of the current sample
//' @param lag_scaling geometric sum for lagged updates
//' @param grad_scaling step size for gradient
//'
//' @return Updates weights and lag.
inline void LaggedUpdate(const unsigned             k,
                         std::vector<double>&       w,
                         const unsigned             n_features,
                         const std::vector<double>& g_sum,
                         std::vector<unsigned>&     lag,
                         const Eigen::MatrixXd&     x,
                         const unsigned             s_ind,
                         const std::vector<double>& lag_scaling,
                         const double               grad_scaling) {

  for (unsigned ind = 0; ind < n_features; ++ind) {
    const unsigned lagged_amount = k - lag[ind];

    if (lagged_amount == 0)
      continue;

    lag[ind] = k;

    w[ind] += lag_scaling[lagged_amount]*grad_scaling*g_sum[ind];
  }
}

inline void LaggedUpdate(const unsigned                     k,
                         std::vector<double>&               w,
                         const unsigned                     n_features,
                         const std::vector<double>&         g_sum,
                         std::vector<unsigned>&             lag,
                         const Eigen::SparseMatrix<double>& x,
                         const unsigned                     s_ind,
                         const std::vector<double>&         lag_scaling,
                         const double                       grad_scaling) {

  for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it) {
    const unsigned ind = it.index();
    const unsigned lagged_amount = k - lag[ind];

    if (lagged_amount == 0)
      continue;

    lag[ind] = k;

    w[ind] += lag_scaling[lagged_amount]*grad_scaling*g_sum[ind];
  }
}

//' Lagged updates for L1-regularized regression
//'
//' @param k current iteration
//' @param w weights vector
//' @param n_features number of features
//' @param g_sum gradient sum
//' @param lag iteration at which the features were last updated
//' @param x the feature matrix. Sparse or dense Eigen object.
//' @param s_ind the index of the current sample
//' @param lag_scaling geometric sum for lagged updates
//' @param prox_scaling step size for the projection step
//' @param grad_scaling step size for gradient step
//' @param prox pointer to the proximal operator
//'
//' @return Updates weights and lag.
inline void LaggedProjection(const unsigned                       k,
                             std::vector<double>&                 w,
                             const unsigned                       n_features,
                             const std::vector<double>&           g_sum,
                             std::vector<unsigned>&               lag,
                             const Eigen::MatrixXd&               x,
                             const unsigned                       s_ind,
                             const std::vector<double>&           lag_scaling,
                             const double                         prox_scaling,
                             const double                         grad_scaling,
                             const std::unique_ptr<sgdnet::Prox>& prox) {

  for (unsigned ind = 0; ind < n_features; ++ind) {
    const unsigned lagged_amount = k - lag[ind];

    if (lagged_amount == 0)
      continue;

    lag[ind] = k;

    w[ind] += grad_scaling*lag_scaling[lagged_amount]*g_sum[ind];
    w[ind] = prox->Evaluate(w[ind], prox_scaling*lag_scaling[lagged_amount]);
  }
}

inline void LaggedProjection(const unsigned                       k,
                             std::vector<double>&                 w,
                             const unsigned                       n_features,
                             const std::vector<double>&           g_sum,
                             std::vector<unsigned>&               lag,
                             const Eigen::SparseMatrix<double>&   x,
                             const unsigned                       s_ind,
                             const std::vector<double>&           lag_scaling,
                             const double                         prox_scaling,
                             const double                         grad_scaling,
                             const std::unique_ptr<sgdnet::Prox>& prox)  {

  for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it) {
    const unsigned ind = it.index();
    const unsigned lagged_amount = k - lag[ind];

    if (lagged_amount == 0)
      continue;

    lag[ind] = k;

    w[ind] += grad_scaling*lag_scaling[lagged_amount]*g_sum[ind];
    w[ind] = prox->Evaluate(w[ind], prox_scaling*lag_scaling[lagged_amount]);
  }
}

//' Dot product
//'
//' @param w weights vector
//' @param n_features number of features
//' @param x the feature matrix. Sparse or dense Eigen object.
//' @param s_ind the index of the current sample
//'
//' @return Returns the dot product of a sample in x with w, used for
//'   predicting the response in a sample.
inline double DotProduct(const std::vector<double>& w,
                         const unsigned             n_features,
                         const unsigned             s_ind,
                         const Eigen::MatrixXd&     x)  {
  double inner_product = 0.0;
  for (unsigned f_ind = 0; f_ind < n_features; ++f_ind)
    inner_product += x(f_ind, s_ind) * w[f_ind];

  return inner_product;
}

inline double DotProduct(const std::vector<double>&         w,
                         const unsigned                     n_features,
                         const unsigned                     s_ind,
                         const Eigen::SparseMatrix<double>& x)  {
  double inner_product = 0.0;
  for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it)
    inner_product += it.value() * w[it.index()];

  return inner_product;
}


//' Weighted addition
//'
//' Updates `y` with a weighted sample in `x`
//'
//' @param y weights or gradient vector
//' @param x the feature matrix. Sparse or dense Eigen object.
//' @param scaling step size
//'
//' @return Updates `y` with `x` scaled.
inline void AddWeighted(std::vector<double>&   y,
                        const Eigen::MatrixXd& x,
                        const unsigned         s_ind,
                        const double           scaling)  {

  for (unsigned f_ind = 0; f_ind < y.size(); ++f_ind)
    y[f_ind] += scaling * x(f_ind, s_ind);
}

inline void AddWeighted(std::vector<double>&               y,
                        const Eigen::SparseMatrix<double>& x,
                        const unsigned                     s_ind,
                        const double                       scaling) {

  for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it)
    y[it.index()] += scaling * it.value();
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
inline void Reset(const unsigned                 k,
                  std::vector<double>&           w,
                  std::vector<double>&           g_sum,
                  std::vector<double>&           lag_scaling,
                  std::vector<unsigned>&         lag,
                  const unsigned                 n_features,
                  const double                   wscale,
                  const double                   prox_scaling,
                  const double                   grad_scaling,
                  const bool                     nontrivial_prox,
                  const std::unique_ptr<sgdnet::Prox>& prox) {
  for (unsigned ind = 0; ind < n_features; ++ind) {
    unsigned lagged_amount = k - lag[ind];

    w[ind] += lag_scaling[lagged_amount]*grad_scaling*g_sum[ind];

    if (nontrivial_prox)
      w[ind] = prox->Evaluate(w[ind], lag_scaling[lagged_amount]*prox_scaling);

    // Rescale weights
    w[ind] *= wscale;
  }
}

//' Lagged updates for L1-regularized regression
//'
//' @param k current iteration
//' @param w weights vector
//' @param n_features number of features
//' @param g_sum gradient sum
//' @param lag iteration at which the features were last updated
//' @param x the feature matrix. Sparse or dense Eigen object.
//' @param s_ind the index of the current sample
//' @param lag_scaling geometric sum for lagged updates
//' @param prox_scaling step size for the projection step
//' @param grad_scaling step size for gradient step
//' @param prox pointer to the proximal operator
//'
//' @return Updates weights and lag.
template <typename T>
void Saga(const T&                               x,
          const std::vector<double>&             y,
          const bool                             fit_intercept,
          const double                           intercept_decay,
          std::vector<double>&                   intercept,
          std::vector<double>&                   w,
          const std::unique_ptr<sgdnet::Family>& family,
          const std::unique_ptr<sgdnet::Prox>&   prox,
          const double                           gamma,
          const double                           alpha,
          const double                           beta,
          std::vector<double>&                   g_sum,
          std::vector<double>&                   g_sum_intercept,
          std::vector<double>&                   g,
          const unsigned                         n_samples,
          const unsigned                         n_features,
          const unsigned                         n_classes,
          const unsigned                         max_iter,
          const double                           tol,
          unsigned&                              n_iter,
          std::vector<unsigned>&                 return_codes,
          std::vector<double>&                   losses,
          const bool                             debug) {

  using namespace std;

  // Are we dealing with a nontrivial prox?
  const bool nontrivial_prox = beta > 0.0;

  // Keep track of when each feature was last updated
  vector<unsigned> lag(n_features);

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

  // Store previous weights for computing stopping criteria
  vector<double> w_previous(w);

  // Outer loop
  unsigned it_outer = 0;
  bool converged = false;
  do {
    // Inner loop
    for (unsigned it_inner = 0; it_inner < n_samples; ++it_inner) {

      // Rescale and unlag weights whenever wscale becomes too small
      if (wscale < sgdnet::SMALL) {
        Reset(it_inner,
              w,
              g_sum,
              lag_scaling,
              lag,
              n_features,
              wscale,
              beta*gamma / wscale,
              -gamma/wscale,
              nontrivial_prox,
              prox);
        wscale = 1.0;
        lag.assign(lag.size(), it_inner);
      }

      // Pull a sample
      unsigned s_ind = floor(R::runif(0.0, n_samples));

      // Apply missed updates to coefficients just-in-time
      if (nontrivial_prox) {
        LaggedProjection(it_inner,
                         w,
                         n_features,
                         g_sum,
                         lag,
                         x,
                         s_ind,
                         lag_scaling,
                         beta*gamma / wscale,
                         -gamma / wscale,
                         prox);
      } else {
        LaggedUpdate(it_inner,
                     w,
                     n_features,
                     g_sum,
                     lag,
                     x,
                     s_ind,
                     lag_scaling,
                     -gamma / wscale);
      }

      double prediction =
        wscale * DotProduct(w, n_features, s_ind, x) + intercept[0];

      double g_new = family->Gradient(prediction, y[s_ind]);
      double g_change = g_new - g[s_ind];
      g[s_ind] = g_new;

      wscale *= wscale_update;

      // Update coefficients (w) with sparse step (with L2 scaling)
      AddWeighted(w, x, s_ind, -g_change*gamma/wscale);

      // TODO(jolars): modify to work with multivariate outcomes
      if (fit_intercept) {
        g_sum_intercept[0] += g_change/n_samples;
        intercept[0] -=
          gamma*g_sum_intercept[0]*intercept_decay + g_change/n_samples;
      }

      // Gradient-average step
      if (nontrivial_prox) {
        LaggedProjection(it_inner + 1,
                         w,
                         n_features,
                         g_sum,
                         lag,
                         x,
                         s_ind,
                         lag_scaling,
                         beta*gamma / wscale,
                         -gamma / wscale,
                         prox);
      } else {
        LaggedUpdate(it_inner + 1,
                     w,
                     n_features,
                     g_sum,
                     lag,
                     x,
                     s_ind,
                     lag_scaling,
                     -gamma / wscale);
      }

      // Update the gradient average
      AddWeighted(g_sum, x, s_ind, g_change/n_samples);

    } // Outer loop

    // Unlag and rescale coefficients
    Reset(n_samples,
          w,
          g_sum,
          lag_scaling,
          lag,
          n_features,
          wscale,
          beta*gamma / wscale,
          -gamma/wscale,
          nontrivial_prox,
          prox);
    wscale = 1.0;
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

