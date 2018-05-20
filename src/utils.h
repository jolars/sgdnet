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

//' Computes the squared norm over rows
//'
//' @param x arma::mat or arma::sp_mat
//'
//' @return The maximum rowise sum of x.
template <typename T>
double RowNormsMax(const T& x) {
  return arma::sum(arma::square(x), 1).max();
}

//' Retrieve step size for the chosen family and feature matrix X
//'
//' @param max_squared_sum the maximum squared rowise norm from X
//' @param alpha_scaled alpha/n_samples
//' @param family the model family
//' @param fit_intercept should an intercept be added to the model?
//' @param n_samples the number of samples in X
//'
//' @return Step size.
//'
//' @keywords internal
double GetStepSize(double         max_squared_sum,
                   double         alpha_scaled,
                   sgdnet::Family family,
                   bool           fit_intercept,
                   arma::uword    n_samples) {
  double L;

  switch(family) {
    case sgdnet::GAUSSIAN: L = max_squared_sum + fit_intercept + alpha_scaled;
                           break;
  }

  return 1.0 / (2.0*L + std::min(2.0*n_samples*alpha_scaled, L));
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
arma::uvec NonzerosInRow(const arma::sp_mat& x, arma::uword i) {

  arma::sp_mat::const_row_iterator x_itr = x.begin_row(i);
  arma::sp_mat::const_row_iterator x_end = x.end_row(i);

  std::vector<unsigned int> out;

  for(; x_itr != x_end; ++x_itr) {
    out.push_back(x_itr.col());
  }

  return arma::conv_to< arma::uvec >::from(out);
}

// For dense matrices, just return a sequence of integers
// TODO: consider using a conditional for sparse matrices to avoid
//       entering this function.
arma::uvec NonzerosInRow(const arma::mat& x, arma::uword i) {
  return arma::regspace<arma::uvec>(0, x.n_cols - 1);
}

//' Preprocess data
//'
//' @param x feature matrix, sparse or dense
//' @param y targets
//' @param normalize whether to normalize x
//' @param fit_intercept wheter to fit the intercept. If false, no
//'   scaling or centering is done.
//' @param x_offset a vector of offsets for each feature
//' @param x_scale a vector of scaling factors (l2 norms) for each vector
//' @param y_offset a vector of means for each column in y
//'
//' @return Nothing. x and y are scaled and centered.
//' @keywords internal
template <typename T>
void Preprocess(T&            x,
                arma::mat&    y,
                const bool    normalize,
                const bool    fit_intercept,
                arma::rowvec& x_offset,
                arma::rowvec& x_scale,
                arma::rowvec& y_offset) {

  // TODO: what is the reason for not scaling and centering when the intercept
  //       is not fit?

  if (fit_intercept) {

    arma::uword n_features = x.n_cols;

    // Center feature matrix
    x_offset = arma::mean(x);
    for (arma::uword i = 0; i < n_features; ++i)
      x.col(i) -= x_offset(i);

    if (normalize) {
      // Normalize each feature with l2-norm
      for (arma::uword i = 0; i < n_features; ++i) {
        x_scale(i) = arma::norm(x.col(i));
        x.col(i) /= x_scale(i);
      }
    } else {
      x_scale.ones();
    }

    // Center targets
    y_offset = arma::mean(y);
    for (arma::uword i = 0; i < y.n_cols; ++i)
      y.col(i) -= y_offset(i);

  } else {
    x_offset.zeros();
    x_scale.ones();
    y_offset.zeros();
  }
}

#endif // SGDNET_UTILS_

