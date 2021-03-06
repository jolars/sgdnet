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

//' Compute step size for SAGA
//'
//' @param max_squared_sum the maximum of the squared norm over samples
//' @param alpha the amount of L1 regularization
//' @param fit_intercept whether the intercept should be fit
//' @param L_scaling scaling factor for the lipschitz constant (L)
//' @param n_samples number of samples
std::vector<double>
StepSize(const double               max_squared_sum,
         const std::vector<double>& alpha,
         const bool                 fit_intercept,
         const double               L_scaling,
         const unsigned             n_samples)
{
  // Lipschitz constant approximation
  std::vector<double> step_sizes;
  step_sizes.reserve(alpha.size());

  for (auto alpha_i : alpha) {
    double L =
      (max_squared_sum + static_cast<double>(fit_intercept))*L_scaling
      + alpha_i;

    double mu_n = 2.0*n_samples*alpha_i;
    step_sizes.emplace_back(1.0 / (2.0*L + std::min(L, mu_n)));
  }
  return step_sizes;
}

//' Computes the squared norm over samples
//'
//' The samples are expected to be in columns here.
//'
//' @param x A Eigen matrix of sparse or dense type.
//'
//' @return The maximum squared norm over columns (samples).
double
ColNormsMax(const Eigen::SparseMatrix<double>& x,
            const Eigen::ArrayXd&              x_center_scaled,
            const bool                         standardize)
{
  auto m = x.cols();

  double norm_max = 0.0;
  for (decltype(m) j = 0; j < m; ++j) {
    double norm =
      standardize ? (x.col(j) - x_center_scaled.matrix()).squaredNorm()
                  : x.col(j).squaredNorm();

    norm_max = std::max(norm_max, norm);
  }

  return norm_max;
}

double
ColNormsMax(const Eigen::MatrixXd& x,
            const Eigen::ArrayXd&  x_center_scaled,
            const bool             standardize) {
  // dense features are already scaled and centered (if required)
  return x.colwise().squaredNorm().maxCoeff();
}

//' Preprocess data
//'
//' Note that we expect samples to have been stored by rows at the point
//' of calling this function
//'
//' @param x feature matrix, sparse or dense
//' @param x_center a vector of offsets for each feature
//' @param x_scale a vector of scaling factors (l2 norms) for each vector
//' @param x_mod mean/stddev to be used for in-place standardization in the
//'   sparse implementation
//'
//' @return Modifies `x`, (possibly) `x_center`, and `x_scale`.
void
PreprocessFeatures(Eigen::MatrixXd& x,
                   Eigen::ArrayXd&  x_center,
                   Eigen::ArrayXd&  x_scale)
{
  x_center = Mean(x);
  x_scale = StandardDeviation(x, x_center);

  Standardize(x, x_center, x_scale);
}

void
PreprocessFeatures(Eigen::SparseMatrix<double>& x,
                   Eigen::ArrayXd&              x_center,
                   Eigen::ArrayXd&              x_scale)
{
  x_center = Mean(x);
  x_scale = StandardDeviation(x, x_center);

  for (decltype(x.cols()) j = 0; j < x.cols(); ++j)
    for (Eigen::SparseMatrix<double>::InnerIterator it(x, j); it; ++it)
      it.valueRef() /= x_scale(j);
}

//' Compute Regularization Path
//'
//' This function computes the regularization path as in glmnet so that
//' the first solution is the null solution (if elasticnet_mix != 0).
//'
//' @param lambda lambda values in input -- empty by default
//' @param n_lambda required number of penalties
//' @param n_classes number of classes
//' @param n_samples number of samples
//' @param lambda_min_ratio smallest lambda_min_ratio
//' @param elasticnet_mix ratio of l1 penalty to l2. Same as alpha in glmnet.
//' @param x feature matrix, samples stored in rows
//' @param y response matrix, samples stored in rows
//' @param y_scale scaling factor for the response matrix
//' @param alpha l2-penalty
//' @param beta l1-penalty
//' @param family pointer to respone type family class object
//'
//' @return lambda, alpha and beta are updated.
template <typename T, typename Family>
void
RegularizationPath(std::vector<double>&   lambda,
                   const unsigned         n_lambda,
                   const unsigned         n_classes,
                   const unsigned         n_samples,
                   const double           lambda_min_ratio,
                   const double           elasticnet_mix,
                   const T&               x,
                   const Eigen::MatrixXd& y,
                   const Eigen::ArrayXd&  y_scale,
                   std::vector<double>&   alpha,
                   std::vector<double>&   beta,
                   const Family&          family)
{
  if (lambda.empty()) {
    double lambda_max =
      family.LambdaMax(x, y, y_scale)/std::max(elasticnet_mix, 0.001);

    if (lambda_max != 0.0)
      lambda = LogSpace(lambda_max, lambda_max*lambda_min_ratio, n_lambda);
    else
      lambda.resize(n_lambda, 0.0);
  }

  alpha.reserve(n_lambda);
  beta.reserve(n_lambda);

  double max_scale = y_scale.maxCoeff();

  // The algorithm uses a different penalty construction than
  // glmnet, so convert lambda values to match alpha and beta from scikit-learn.
  for (double& lambda_i : lambda) {
    // Scaled L2 penalty
    alpha.emplace_back((1.0 - elasticnet_mix)*lambda_i/max_scale);
    // Scaled L1 penalty
    beta.emplace_back(elasticnet_mix*lambda_i/max_scale);
    // Rescale lambda so to return to user on the scale of y
  }
}

//' Loss for the current epoch
//'
//' @param x feature matrix
//' @param y response matrix
//' @param w coefficients
//' @param intercept the intercept
//' @param family a Family class object for the current response type
//' @param alpha scaled l2-penalty
//' @param beta scaled l1-penalty
//' @param n_samples number of samples
//' @param n_features number of features
//' @param n_classes number of pseudo-classes
//'
//' @return The loss of the current epoch is appended to `losses`.
//'
//' @noRd
template <typename T, typename Family>
double
EpochLoss(const T&               x,
          const Eigen::ArrayXd&  x_center_scaled,
          const Eigen::MatrixXd& y,
          const Eigen::ArrayXXd& w,
          const Eigen::ArrayXd&  intercept,
          const Family&          family,
          const double           alpha,
          const double           beta,
          const unsigned         n_samples,
          const unsigned         n_features,
          const unsigned         n_classes,
          const bool             is_sparse,
          const bool             standardize)
{
  Eigen::ArrayXd linear_predictor(n_classes);

  double loss = 0.0;

  for (unsigned s_ind = 0; s_ind < n_samples; ++s_ind) {
    linear_predictor = (w.matrix() * x.col(s_ind)).array() + intercept;
    if (standardize && is_sparse)
      linear_predictor -= (w.matrix() * x_center_scaled.matrix()).array();
    loss += family.Loss(linear_predictor, y, s_ind)/n_samples;
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
struct ConvergenceCheck {

  ConvergenceCheck(const Eigen::ArrayXXd& w, const double tol)
                   : w_prev(w), tol(tol) {}

  bool
  operator()(const Eigen::ArrayXXd& w_new)
  {
    double max_change = (w_new - w_prev).abs().maxCoeff();
    double max_size   = w_new.abs().maxCoeff();

    bool all_zero  = (max_size == 0.0) && (max_change == 0.0);
    bool no_change = (max_size != 0.0) && (max_change/max_size <= tol);

    w_prev = w_new;

    return all_zero || no_change;
  }

private:
  Eigen::ArrayXXd w_prev;
  const double    tol;
};

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
inline
void
AdaptiveTranspose(Eigen::SparseMatrix<double>& x)
{
  x = x.transpose().eval();
}

inline
void
AdaptiveTranspose(Eigen::MatrixXd& x)
{
  x.transposeInPlace();
}

//' Deviance
//'
//' Computes the deviance of the model given by `weights` and `intercept`.
//'
//' @param x a feature matrix (dense or sparse), samples in columns
//' @param weights a vector of coefficients
//' @param intercept an intercept vector
//' @param is_sparse whether x is sparse
//' @param n_samples the number of samples
//' @param n_feature the number of features
//' @param n_classes the number of classes
//' @param family a pointer to the family object
//'
//' @return Returns the deviance.
template <typename T, typename Family>
double
Deviance(const T&               x,
         const Eigen::ArrayXd&  x_center_scaled,
         const Eigen::MatrixXd& y,
         const Eigen::ArrayXXd& w,
         const Eigen::ArrayXd&  intercept,
         const unsigned         n_samples,
         const unsigned         n_features,
         const unsigned         n_classes,
         const Family&          family,
         const bool             is_sparse,
         const bool             standardize)
{
  double loss = 0.0;
  Eigen::ArrayXd linear_predictor(n_classes);

  for (unsigned s_ind = 0; s_ind < n_samples; ++s_ind) {
    linear_predictor = (w.matrix() * x.col(s_ind)).array() + intercept;
    if (standardize && is_sparse)
      linear_predictor -= (w.matrix() * x_center_scaled.matrix()).array();
    loss += family.Loss(linear_predictor, y, s_ind);
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
void
Rescale(Eigen::ArrayXXd               weights,
        std::vector<Eigen::ArrayXXd>& weights_archive,
        Eigen::ArrayXd                intercept,
        std::vector<Eigen::ArrayXd>&  intercept_archive,
        const Eigen::ArrayXd&         x_center,
        const Eigen::ArrayXd&         x_scale,
        const Eigen::ArrayXd&         y_center,
        const Eigen::ArrayXd&         y_scale,
        const bool                    fit_intercept)
{
  auto m = weights.cols();
  auto p = weights.rows();

  Eigen::ArrayXd x_bar_beta_sum = Eigen::ArrayXd::Zero(p);

  for (decltype(p) j = 0; j < m; ++j) {
    weights.col(j) *= y_scale/x_scale(j);
    x_bar_beta_sum += x_center(j)*weights.col(j);
  }

  if (fit_intercept)
    intercept = intercept*y_scale + y_center - x_bar_beta_sum;

  weights_archive.emplace_back(std::move(weights));
  intercept_archive.emplace_back(std::move(intercept));
}

#endif // SGDNET_UTILS_

