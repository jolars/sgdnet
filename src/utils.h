// sgdnet: Penalized Generalized Linear Models with Stochastic Gradient Descent
// Copyright (C) 2018 Johan Larsson
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

#ifndef SGDNET_UTILS_
#define SGDNET_UTILS_

#include <RcppArmadillo.h>
#include "families.h"
#include "math.h"

//' Computes the squared norm over rows
//'
//' @param x arma::mat or arma::sp_mat
//'
//' @return The maximum rowise sum of x.
template <typename T>
double ColNormsMax(const T& x) {
  return arma::sum(arma::square(x), 0).max();
}

//' Return index of nonzero elements
//'
//' The function is overloaded to work with both dense and sparse matrices.
//'
//' @param x_it iterator
//' @param i counter
//'
//' @return Index of iterator in current row or column.
//'
//' @keywords internal
std::vector<std::size_t> Nonzeros(const arma::SpSubview<double> x) {

  arma::SpSubview<double>::const_iterator x_itr = x.begin();
  arma::SpSubview<double>::const_iterator x_end = x.end();

  std::vector<std::size_t> out;

  for(; x_itr != x_end; ++x_itr)
    out.push_back(x_itr.row());

  return out;
}

// For dense matrices, just return a sequence of integers. This function
// should never get called.
// TODO: consider using a conditional for sparse matrices to avoid
//       entering this function.
std::vector<std::size_t> Nonzeros(const arma::subview<double> x) {
  std::vector<std::size_t> out(x.n_elem);
  std::iota(out.begin(), out.end(), 0);
  return out;
}

//' Preprocess data
//'
//' @param x feature matrix, sparse or dense
//' @param y targets
//' @param normalize whether to normalize x
//' @param fit_intercept wheter to fit the intercept. If false, no
//'   scaling or centering is done.
//' @param x_center a vector of offsets for each feature
//' @param x_scale a vector of scaling factors (l2 norms) for each vector
//' @param y_center a vector of means for each column in y
//'
//' @return Nothing. x and y are scaled and centered.
//' @keywords internal
template <typename T>
void PreprocessFeatures(T&                   x,
                        const bool           normalize,
                        const bool           fit_intercept,
                        std::vector<double>& x_center,
                        std::vector<double>& x_scale,
                        const bool           is_sparse,
                        const std::size_t    n_features,
                        const std::size_t    n_samples) {

  // TODO: what is the reason for not scaling and centering when the intercept
  //       is not fit?

  if (fit_intercept) {
    // Center feature matrix with mean
    for (std::size_t feature_ind = 0; feature_ind < n_features; ++feature_ind) {

      double x_col_mu = Mean(x.col(feature_ind), n_samples);

      if (is_sparse) {
        x_center.push_back(0.0);
      } else {
        x_center.push_back(x_col_mu);
        x.col(feature_ind) -= x_col_mu;
      }

      if (normalize) {
        double x_col_sd = StandardDeviation(x.col(feature_ind), n_samples);

        x_scale.push_back(x_col_sd);

        if (x_col_sd != 0.0)
          x.col(feature_ind) /= x_col_sd;

      } else {
        x_scale.push_back(1.0);
      }
    }
  } else {
    std::fill_n(x_center.begin(), n_features, 0.0);
    std::fill_n(x_scale.begin(), n_features, 1.0);
  }
}

// TODO(jolars): move this into the Family class
template <typename T>
inline double LambdaMax(const T&                   x,
                        const std::vector<double>& y,
                        const std::size_t          n_samples,
                        const double               elasticnet_mix) {
  arma::rowvec yt(y);

  // Cap elasticnet_mix (alpha in glmnet) to 0.001
  return
    arma::abs(yt * x.t()).max()/(n_samples*std::max(elasticnet_mix, 0.001));
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
void PredictSample(std::vector<double>&            prediction,
                   const T&                        x,
                   const std::vector<std::size_t>& nonzero_indices,
                   const std::vector<double>&      weights,
                   const double                    wscale,
                   const std::vector<double>&      intercept,
                   const std::size_t               n_classes) {

  for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind) {
    double inner_product = 0.0;
    auto x_itr = x.begin();
    for (const auto& feature_ind : nonzero_indices) {
      inner_product += weights[feature_ind*n_classes + class_ind] * (*x_itr);
      ++x_itr;
    }

    prediction[class_ind] = wscale*inner_product + intercept[class_ind];
  }
}

template <typename T>
void EpochLoss(const T&                         x,
               const std::vector<double>&       y,
               const std::vector<double>&       weights,
               const std::vector<double>&       intercept,
               std::unique_ptr<sgdnet::Family>& family,
               const double                     alpha_scaled,
               const double                     beta_scaled,
               const std::size_t                n_samples,
               const std::size_t                n_classes,
               const bool                       is_sparse,
               std::vector<double>&             losses) {

  double loss = 0.0;
  double l1_norm = 0.0;
  double l2_norm_squared = 0.0;

  for (const auto& weight : weights) {
    l1_norm += std::abs(weight);
    l2_norm_squared += weight*weight;
  }

  loss += 0.5*l2_norm_squared*alpha_scaled*n_samples
    + l1_norm*beta_scaled*n_samples;

  std::vector<std::size_t> nonzero_indices = Nonzeros(x.col(0));

  for (std::size_t sample_ind = 0; sample_ind < n_samples; ++sample_ind) {
    if (is_sparse && sample_ind > 0)
      nonzero_indices = Nonzeros(x.col(sample_ind));

    for (std::size_t class_ind = 0; class_ind < n_classes; ++class_ind) {
      double inner_product = 0.0;
      auto x_itr = x.begin_col(sample_ind);
      for (const auto& feature_ind : nonzero_indices) {
        inner_product += (*x_itr)*weights[feature_ind*n_classes + class_ind];
        ++x_itr;
      }
      loss += family->Loss(inner_product + intercept[class_ind],
                           y[sample_ind*n_classes + class_ind]);
    }
  }
  losses.push_back(loss);
}

#endif // SGDNET_UTILS_

