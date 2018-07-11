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

#include <RcppEigen.h>
#include "families.h"
#include "math.h"

//' Computes the squared norm over samples
//'
//' These samples are expected to be in columns here.
//'
//' @param x A Eigen matrix of sparse or dense type.
//'
//' @return The maximum squared norm over columns (samples).
inline double ColNormsMax(const Eigen::MatrixXd& x) {
  double cn_max = 0.0;

  for (unsigned i = 0; i < x.cols(); ++i) {

    double cn_current = 0.0;
    for (unsigned j = 0; j < x.rows(); ++j)
      cn_current += x(i, j)*x(i, j);

    cn_max = std::max(cn_max, cn_current);
  }

  return cn_max;
}

inline double ColNormsMax(const Eigen::SparseMatrix<double>& x) {
  double cn_max = 0.0;

  for (unsigned i = 0; i < x.outerSize(); ++i) {

    double cn_current = 0.0;
    for (Eigen::SparseMatrix<double>::InnerIterator it(x, i); it; ++it)
      cn_current += it.value()*it.value();

    cn_max = std::max(cn_max, cn_current);
  }

  return cn_max;
}

//' Preprocess data
//'
//' @param x feature matrix, sparse or dense
//' @param x_center a vector of offsets for each feature
//' @param x_scale a vector of scaling factors (l2 norms) for each vector
//'
//' @return Modifies `x_center` and `x_scale`.
void PreprocessFeatures(Eigen::MatrixXd&     x,
                        std::vector<double>& x_center,
                        std::vector<double>& x_scale,
                        const unsigned       n_features,
                        const unsigned       n_samples) {

  // Center feature matrix with mean

  for (unsigned f_ind = 0; f_ind < n_features; ++f_ind) {

    double x_col_mu = x.col(f_ind).sum()/n_samples;

    x_center[f_ind] = x_col_mu;

    double var = 0.0;
    for (unsigned s_ind = 0; s_ind < n_samples; ++s_ind) {
      x.coeffRef(s_ind, f_ind) -= x_col_mu;
      var += x(s_ind, f_ind)*x(s_ind, f_ind)/n_samples;
    }

    double x_col_sd = std::sqrt(var);
    if (x_col_sd == 0.0) x_col_sd = 1.0;

    for (unsigned s_ind = 0; s_ind < n_samples; ++s_ind)
      x.coeffRef(s_ind, f_ind) /= x_col_sd;

    x_scale[f_ind] = x_col_sd;
  }
}

void PreprocessFeatures(Eigen::SparseMatrix<double>& x,
                        std::vector<double>&         x_center,
                        std::vector<double>&         x_scale,
                        const unsigned               n_features,
                        const unsigned               n_samples) {

  // Center feature matrix with mean
  for (unsigned f_ind = 0; f_ind < n_features; ++f_ind) {

    double x_col_mu = x.col(f_ind).sum()/n_samples;

    // no centering for sparse matrices
    x_center[f_ind] = 0.0;

    double var = 0.0;
    for (Eigen::SparseMatrix<double>::InnerIterator it(x, f_ind); it; ++it)
      var += std::pow(it.value() - x_col_mu, 2)/n_samples;

    double x_col_sd = std::sqrt(var);
    if (x_col_sd == 0.0) x_col_sd = 1.0;

    for (Eigen::SparseMatrix<double>::InnerIterator it(x, f_ind); it; ++it)
      it.valueRef() /= x_col_sd;

    x_scale[f_ind] = x_col_sd;
  }
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
//' @param family pointer to respone type family class object
//'
//' @return lambda, alpha and beta are updated.
template <typename T>
void RegularizationPath(std::vector<double>&                   lambda,
                        const unsigned                         n_lambda,
                        const double                           lambda_min_ratio,
                        const double                           elasticnet_mix,
                        const T&                               x,
                        std::vector<double>&                   alpha,
                        std::vector<double>&                   beta,
                        const std::unique_ptr<sgdnet::Family>& family) {
  double lambda_scaling = family->lambda_scaling;

  if (lambda.empty()) {
    double lambda_max = family->LambdaMax(x);
    lambda_max /= std::max(elasticnet_mix, 0.001);
    if (lambda_max != 0.0)
      lambda = LogSpace(lambda_max, lambda_max*lambda_min_ratio, n_lambda);
    else
      lambda.resize(n_lambda, 0.0);
  }

  alpha.reserve(n_lambda);
  beta.reserve(n_lambda);

  // The algorithm uses a different penalty construction than
  // glmnet, so convert lambda values to match alpha and beta from scikit-learn.
  for (double& lambda_i : lambda) {
    // Scaled L2 penalty
    alpha.emplace_back((1.0 - elasticnet_mix)*lambda_i/lambda_scaling);
    // Scaled L1 penalty
    beta.emplace_back(elasticnet_mix*lambda_i/lambda_scaling);
    // Rescale lambda so to return to user on the scale of y
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
inline void PredictSample(std::vector<double>&       prediction,
                          const std::vector<double>& w,
                          const double               wscale,
                          const unsigned             n_features,
                          const unsigned             n_classes,
                          const unsigned             s_ind,
                          const Eigen::MatrixXd&     x,
                          const std::vector<double>& intercept) {
  for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind) {
    double inner_product = 0.0;
    for (unsigned f_ind = 0; f_ind < n_features; ++f_ind)
      inner_product += x(f_ind, s_ind) * w[f_ind*n_classes + c_ind];

    prediction[c_ind] = wscale*inner_product + intercept[c_ind];
  }
}

inline void PredictSample(std::vector<double>&               prediction,
                          const std::vector<double>&         w,
                          const double                       wscale,
                          const unsigned                     n_features,
                          const unsigned                     n_classes,
                          const unsigned                     s_ind,
                          const Eigen::SparseMatrix<double>& x,
                          const std::vector<double>&         intercept) {
  for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind) {
    double inner_product = 0.0;
    for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it)
      inner_product += it.value() * w[it.index()*n_classes + c_ind];

    prediction[c_ind] = wscale*inner_product + intercept[c_ind];
  }
}

//' Loss for the current epoch
//'
//' @param x feature matrix
//' @param y response vector
//' @param weights coefficients
//' @param intercept the intercept
//' @param family a Family class object for the current response type
//' @param alpha scaled l2-penalty
//' @param beta scaled l1-penalty
//' @param n_samples number of samples
//' @param n_features number of features
//' @param n_classes number of pseudo-classes
//' @param losses loss vector, which the current loss vector will be
//'   appended to
//'
//' @return The loss of the current epoch is appended to `losses`.
//'
//' @noRd
template <typename T>
double EpochLoss(const T&                               x,
                 const std::vector<double>&             w,
                 const std::vector<double>&             intercept,
                 const std::unique_ptr<sgdnet::Family>& family,
                 const double                           alpha,
                 const double                           beta,
                 const unsigned                         n_samples,
                 const unsigned                         n_features,
                 const unsigned                         n_classes) {

  double loss = 0.0;
  double l1_norm = 0.0;
  double l2_norm_squared = 0.0;

  for (auto w_i : w) {
    l1_norm += std::abs(w_i);
    l2_norm_squared += w_i*w_i;
  }

  std::vector<double> prediction(n_classes);

  for (unsigned s_ind = 0; s_ind < n_samples; ++s_ind) {
    PredictSample(prediction,
                  w,
                  1.0,
                  n_features,
                  n_classes,
                  s_ind,
                  x,
                  intercept);

    loss += family->Loss(prediction, s_ind)/n_samples;
  }

  return loss;
}

//' Convergence check for SAGA algorithm
//'
//' Returns true, indicating that the algorithm converged, if the
//' absolute of the largest relative change surpasses `tol`.
//'
//' @param w weights
//' @param w_previous previous weights
//' @param tol tolerance (threshold) for convergence, compared
//'   against the largest relative change.
//'
//' @return `true` if the algorithm converged, `false` otherwise.
bool CheckConvergence(const std::vector<double>& w,
                      std::vector<double>&       w_previous,
                      const double               tol) {

  double max_change = 0.0;
  double max_weight = 0.0;

  for (unsigned i = 0; i < w.size(); ++i) {
    max_weight = std::max(max_weight, std::abs(w[i]));
    max_change = std::max(max_change, std::abs(w[i] - w_previous[i]));
    w_previous[i] = w[i];
  }

  bool all_zero  = (max_weight == 0.0) && (max_change == 0.0);
  bool no_change = (max_weight != 0.0) && (max_change/max_weight <= tol);

  return all_zero || no_change;
}

//' Adapative transposing of feature matrix
//'
//' For sparse matrices, Eigen (yet?) have a inplace
//' transpose method, so we overload for sparse and dense matrices,
//' transposing inplace when x is dense.
//'
//' @param x a sparse or dense matrix
//'
//' @return x transposed.
//'
//' @keywords internal
//' @noRd
inline void AdaptiveTranspose(Eigen::SparseMatrix<double>& x) {
  x = x.transpose().eval();
}

inline void AdaptiveTranspose(Eigen::MatrixXd& x) {
  x.transposeInPlace();
}

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
//' @param family a pointer to the family object
//'
//' @return Returns the deviance.
template <typename T>
double Deviance(const T&                           x,
                const std::vector<double>&         w,
                const std::vector<double>&         intercept,
                const unsigned                     n_samples,
                const unsigned                     n_features,
                const unsigned                     n_classes,
                std::unique_ptr<sgdnet::Family>&   family) {
  double loss = 0.0;
  std::vector<double> prediction(n_classes);

  for (unsigned s_ind = 0; s_ind < n_samples; ++s_ind) {
    PredictSample(prediction,
                  w,
                  1.0,
                  n_features,
                  n_classes,
                  s_ind,
                  x,
                  intercept);

    loss += family->Loss(prediction, s_ind);
  }
  return 2.0 * loss;
}

//' Rescale and store weights and intercept
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
//' @param fit_intercept whether to fit the intercept or not
//'
//' @return `weights` and `intercept` are rescaled and stored in weights_archive
//'   and intercept_archive.
void Rescale(std::vector<double>               weights,
             std::vector<std::vector<double>>& weights_archive,
             std::vector<double>               intercept,
             std::vector<std::vector<double>>& intercept_archive,
             const std::vector<double>&        x_center,
             const std::vector<double>&        x_scale,
             const std::vector<double>&        y_center,
             const std::vector<double>&        y_scale,
             const unsigned                    n_features,
             const unsigned                    n_classes,
             const bool                        fit_intercept) {

  std::vector<double> x_scale_prod(n_classes);
  for (unsigned f_ind = 0; f_ind < n_features; ++f_ind) {
    for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind) {
      unsigned idx = f_ind*n_classes + c_ind;
      weights[idx] *= y_scale[c_ind]/x_scale[f_ind];
      x_scale_prod[c_ind] += x_center[f_ind]*weights[idx];
    }
  }

  if (fit_intercept) {
    for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind)
      intercept[c_ind] =
        intercept[c_ind]*y_scale[c_ind] + y_center[c_ind] - x_scale_prod[c_ind];
  }

  weights_archive.push_back(std::move(weights));
  intercept_archive.push_back(std::move(intercept));
}

#endif // SGDNET_UTILS_

