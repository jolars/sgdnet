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

#ifndef SGDNET_SAGA_SPARSE_
#define SGDNET_SAGA_SPARSE_

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
//' @param subx the selected feature matrix. Sparse or dense Eigen object.
//' @param lag_scaling geometric sum for lagged updates
//' @param wscale the current scale of the coefficients
//' @param penalty object of Penalty class
//'
//' @return Updates weights and lag.
template <typename Penalty>
inline
void
LaggedUpdate(const unsigned                     k,
             Eigen::ArrayXXd&                   w,
             const unsigned                     n_features,
             const Eigen::ArrayXXd&             g_sum,
             std::vector<unsigned>&             lag,
             const Eigen::SparseMatrix<double>& subx,
             const std::vector<double>&         lag_scaling,
             const double                       wscale,
             const Penalty&                     penalty) noexcept
{
  for (unsigned m = 0; m < subx.cols(); ++m) {

    for (Eigen::SparseMatrix<double>::InnerIterator it(subx, m); it; ++it) {

      auto j = it.index();
      auto lagged_amount = k - lag[j];

      if (lagged_amount != 0) {
        penalty(w, j, wscale, lag_scaling[lagged_amount], g_sum);
        lag[j] = k;
      }
    }
  }
}

//' Weighted addition
//'
//' Updates `a` with a weighted sample in `x`
//'
//' @param a weights or gradient vector
//' @param subx the selected feature matrix. Sparse or dense Eigen object.
//' @param B the batchsize
//' @param n_classes number of classes
//' @param g_change change in gradient
//' @param scaling step size
//'
//' @return Updates `a` with `x` scaled.
inline
void
AddWeighted(Eigen::ArrayXXd&                   a,
            const Eigen::SparseMatrix<double>& subx,
            const Eigen::ArrayXd&              x_center_scaled,
            const unsigned                     B,
            const unsigned                     n_classes,
            const Eigen::ArrayXXd&             g_change,
            const double                       scaling,
            const bool                         standardize) noexcept
{
  Eigen::ArrayXd g_change_col = Eigen::ArrayXd::Zero(n_classes);
  for (decltype(a.rows()) k = 0; k < a.rows(); ++k) {

    for (unsigned i = 0; i < B; ++i) {
        g_change_col = g_change.col(i);
        a.row(k) += subx.col(i)*g_change_col(k)*scaling;
        if (standardize)
          a.row(k) -= x_center_scaled*g_change_col(k)*scaling;
    }

  }
}

template <typename Penalty>
inline
double
Reset(const unsigned         k,
      Eigen::ArrayXXd&       w,
      const Eigen::ArrayXXd& g_sum,
      std::vector<double>&   lag_scaling,
      std::vector<unsigned>& lag,
      const unsigned         n_features,
      double                 wscale,
      const Penalty&         penalty) noexcept
{
  for (unsigned j = 0; j < n_features; ++j) {

    auto lagged_amount = k - lag[j];

    if (lagged_amount != 0)
      penalty(w, j, wscale, lag_scaling[lagged_amount], g_sum);
  }

  w *= wscale;

  return 1.0;
}

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
//' @param cyclic whether we use cyclic SAGA or not.
//' @param B the batchsize
//'
//' @return Updates `w`, `intercept`, `g_sum`, `g_sum_intercept`, `g`,
//'   `n_iter`, `return_codes`, and possibly `losses`.
template <typename Family, typename Penalty>
void
Saga(Penalty&                           penalty,
     const Eigen::SparseMatrix<double>& x,
     const Eigen::ArrayXd&              x_center_scaled,
     const Eigen::VectorXd&             weight,
     const Eigen::MatrixXd&             y,
     Eigen::ArrayXd&                    intercept,
     const bool                         fit_intercept,
     const bool                         is_sparse,
     const bool                         standardize,
     Eigen::ArrayXXd&                   w,
     const Family&                      family,
     const double                       gamma,
     const double                       alpha,
     const double                       beta,
     Eigen::ArrayXXd&                   g_memory,
     Eigen::ArrayXXd&                   g_sum,
     Eigen::ArrayXd&                    g_sum_intercept,
     const unsigned                     n_samples,
     const unsigned                     n_features,
     const unsigned                     n_classes,
     const unsigned                     max_iter,
     const double                       tol,
     unsigned&                          n_iter,
     std::vector<unsigned>&             return_codes,
     std::vector<double>&               losses,
     const bool                         debug,
     const bool                         cyclic,
     const unsigned                     B) noexcept
{
  using namespace std;

  // epoch size
  const unsigned epoch = floor(n_samples/B);

  // Keep track of when each feature was last updated
  std::vector<unsigned> lag(n_features);

  double wscale = 1.0;

  vector<double> lag_scaling;
  lag_scaling.reserve(epoch + 1);
  lag_scaling.push_back(0.0);
  lag_scaling.push_back(1.0);
  double geo_sum = 1.0;
  double wscale_update = 1.0 - alpha*gamma;

  for (unsigned i = 2; i < epoch + 1; ++i) {
    geo_sum *= wscale_update;
    double tmp = lag_scaling.back() + geo_sum;
    lag_scaling.push_back(tmp);
  }

  penalty.setParameters(gamma, alpha, beta);

  // Gradient vector and change in gradient vector
  Eigen::ArrayXXd g                = Eigen::ArrayXXd::Zero(n_classes, B);
  Eigen::ArrayXXd g_change         = Eigen::ArrayXXd::Zero(n_classes, B);

  Eigen::ArrayXXd linear_predictor = Eigen::ArrayXXd::Zero(n_classes, B);

  // Setup functor for checking convergence
  ConvergenceCheck convergence_check{w, tol};

  // Setup index generator
  Eigen::ArrayXXi ind   = Eigen::ArrayXXi::Zero(B, epoch);
  Eigen::ArrayXi  s_ind = Eigen::ArrayXi::Zero(B);

  // Setup selected sample matrix
  Eigen::SparseMatrix<double> subx(n_features, B);

  // Outer loop
  unsigned it_outer = 0;
  bool converged = false;
  do {

    // Pull an epoch of samples
    ind = Ind(n_samples, B, cyclic);

    // Inner loop
    for (unsigned it_inner = 0; it_inner < epoch; ++it_inner) {

      // Pull a sample
      s_ind = ind.col(it_inner);

      // Select samples
      subx = SelectSparse(x, s_ind);

      LaggedUpdate(it_inner,
                   w,
                   n_features,
                   g_sum,
                   lag,
                   subx,
                   lag_scaling,
                   wscale,
                   penalty);

      linear_predictor = ((w.matrix() * subx).array()*wscale).colwise() + intercept;

      if (standardize)
        linear_predictor -= (w.matrix() * x_center_scaled.matrix()).array()*wscale;

      family.Gradient(linear_predictor, y, s_ind, weight, g);

      g_change = g - SelectArray(g_memory, s_ind);
      SetCol(g_memory, g, s_ind);

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
      if (fit_intercept) {
        g_sum_intercept += g_change.rowwise().sum()/n_samples;
        // The 0.01 factor is intercept decay to avoid intercept oscillation
        intercept -= gamma*(g_sum_intercept*0.01 +  g_change.rowwise().sum()/n_samples);
      }

      AddWeighted(w,
                  subx,
                  x_center_scaled,
                  B,
                  n_classes,
                  g_change,
                  -(gamma/(wscale*B)),
                  standardize);

      // Gradient-average step
      LaggedUpdate(it_inner + 1,
                   w,
                   n_features,
                   g_sum,
                   lag,
                   subx,
                   lag_scaling,
                   wscale,
                   penalty);

      // Update the gradient average
      AddWeighted(g_sum,
                  subx,
                  x_center_scaled,
                  B,
                  n_classes,
                  g_change,
                  1.0/n_samples,
                  standardize);

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
                              standardize,
                              weight);
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

#endif /* SGDNET_SAGA-SPARSE_ */


