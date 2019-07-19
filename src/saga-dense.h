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

#ifndef SGDNET_SAGA_DENSE_
#define SGDNET_SAGA_DENSE_

#include <RcppEigen.h>
#include "utils.h"
#include "math.h"
#include "families.h"
#include "penalties.h"
#include "prox.h"
#include "constants.h"
#include "prox.h"

//' The SAGA algorithm
//'
//' @param x the feature matrix
//' @param y response vector or vectorized response matrix
//' @param intercept the vector. Initialized to zero but will be stored
//'   and continually updated along the regularization path to support
//'   warm starts
//' @param fit_intercept whether the intercept should be fit
//' @param intercept_decay adjustment of learning rate for intercept,
//'   which is lower for sparse features to guard against intercept
//'   oscillation
//' @param w weights. Updated in the same manner as `intercept`.
//' @param family a pointer to the Family object
//' @param penalty an object of class Penalty
//' @param gamma step size
//' @param alpha L2-regularization penalty strength
//' @param beta L1-regularization penalty strength
//' @param g_memory a storage for gradients
//' @param g_sum gradient sum
//' @param g_sum_intercept gradient sum for intercept
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
template <typename Family, typename Penalty>
void
Saga(Penalty&               penalty,
     const Eigen::MatrixXd& x,
     const Eigen::ArrayXd&  x_center_scaled,
     const Eigen::MatrixXd& y,
     Eigen::ArrayXd&        intercept,
     const bool             fit_intercept,
     const bool             is_sparse,
     const bool             standardize,
     Eigen::ArrayXXd&       w,
     const Family&          family,
     const double           gamma,
     const double           alpha,
     const double           beta,
     const double           noncov,
     Eigen::ArrayXXd&       g_memory,
     Eigen::ArrayXXd&       g_sum,
     Eigen::ArrayXd&        g_sum_intercept,
     const unsigned         n_samples,
     const unsigned         n_features,
     const unsigned         n_classes,
     const unsigned         max_iter,
     const double           tol,
     unsigned&              n_iter,
     std::vector<unsigned>& return_codes,
     std::vector<double>&   losses,
     const bool             debug) noexcept
{
  using namespace std;

  double wscale = 1.0;

  double wscale_update = 1.0 - alpha*gamma;
  if (noncov > 0.0) { wscale_update = 1.0; }

  penalty.setParameters(gamma, alpha, beta, noncov);

  // Gradient vector and change in gradient vector
  Eigen::ArrayXd g        = Eigen::ArrayXd::Zero(n_classes);
  Eigen::ArrayXd g_change = Eigen::ArrayXd::Zero(n_classes);

  Eigen::ArrayXd linear_predictor(n_classes);

  // Setup functor for checking convergence
  ConvergenceCheck convergence_check{w, tol};

  // Outer loop
  unsigned it_outer = 0;
  bool converged = false;
  do {
    // Inner loop
    for (unsigned it_inner = 0; it_inner < n_samples; ++it_inner) {

      // Pull a sample
      unsigned s_ind = floor(R::runif(0.0, n_samples));

      linear_predictor = (w.matrix() * x.col(s_ind)).array()*wscale + intercept;

      family.Gradient(linear_predictor, y, s_ind, g);

      g_change = g - g_memory.col(s_ind);
      g_memory.col(s_ind) = g;

      // Rescale and unlag weights whenever wscale becomes too small
      if (wscale < sgdnet::SMALL) {
        // Unlag and rescale coefficients
        w *= wscale;
        wscale = 1.0;
      }

      wscale *= wscale_update;

      if (fit_intercept) {
        g_sum_intercept += g_change/n_samples;
        intercept -= gamma*(g_sum_intercept + g_change/n_samples);
      }

      // Update coefficients (w) with sparse step (with L2 scaling)
      w -= g_change.rowwise()*x.col(s_ind).transpose().array()*(gamma/wscale);

      // Gradient-average step
      for (unsigned j = 0; j < n_features; ++j)
        penalty(w, j, wscale, 1.0, g_sum);

      // Update the gradient average
      g_sum += g_change.rowwise()*x.col(s_ind).transpose().array()/n_samples;

    } // Outer loop

    // Unlag and rescale coefficients
    w *= wscale;
    wscale = 1.0;

    if (debug) {
      double loss = EpochLoss(x,
                              x_center_scaled,
                              y,
                              w,
                              intercept,
                              family,
                              alpha,
                              beta,
                              n_samples,
                              n_features,
                              n_classes,
                              is_sparse,
                              standardize);
      losses.push_back(loss);
    }

    converged = convergence_check(w);

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

#endif /* SGDNET_SAGA-DENSE_ */



