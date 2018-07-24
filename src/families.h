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

#ifndef SGDNET_FAMILIES_
#define SGDNET_FAMILIES_

#include <memory>
#include "math.h"
#include "utils.h"
namespace sgdnet {

class Family {
public:
  Family(const std::vector<double>& y,
         const unsigned             n_samples,
         const unsigned             n_classes,
         const double               L_scaling)
         : y(y),
           n_samples(n_samples),
           n_classes(n_classes),
           L_scaling(L_scaling),
           y_center(n_classes),
           y_scale(n_classes, 1.0),
           y_mat_scale(n_classes, 1.0),
           gradient_memory(n_classes*n_samples, 0.0) {
    y_mat.setZero(n_samples, n_classes);
  };

  double Loss(const unsigned i);

  void Gradient(const unsigned i);

  void PreprocessResponse();

  double NullDeviance(const bool fit_intercept);

  void Predict(const std::vector<double>& w,
               const double               wscale,
               const unsigned             n_features,
               const unsigned             s_ind,
               const Eigen::MatrixXd&     x);

  void Predict(const std::vector<double>&         w,
               const double                       wscale,
               const unsigned                     n_features,
               const unsigned                     s_ind,
               const Eigen::SparseMatrix<double>& x);

  std::vector<double> StepSize(const double               max_squared_sum,
                               const std::vector<double>& alpha_scaled,
                               const bool                 fit_intercept) {
    // Lipschitz constant approximation
    std::vector<double> step_sizes;
    step_sizes.reserve(alpha_scaled.size());

    for (auto alpha_i : alpha_scaled) {
      double L =
        L_scaling*(max_squared_sum + static_cast<double>(fit_intercept))
        + alpha_i;
      double mu_n = 2.0*n_samples*alpha_i;
      step_sizes.push_back(1.0 / (2.0*L + std::min(L, mu_n)));
    }
    return step_sizes;
  }

  std::vector<double> y;
  std::vector<double> y_center;
  std::vector<double> y_scale;
  Eigen::MatrixXd y_mat;
  std::vector<double> y_mat_scale;
  std::vector<double> gradient_memory;
  unsigned n_samples;
  unsigned n_classes;
  unsigned n_targets;
  double null_deviance = 0.0;
  double L_scaling;
  double lambda_scaling = 1.0;
};

class Gaussian : public Family {
public:
  Gaussian(const std::vector<double>& y,
           const unsigned             n_samples,
           const unsigned             n_classes)
           : Family(y, n_samples, n_classes, 1.0) {}

  void PreprocessResponse() {
    y_center[0] = Mean(y);
    y_scale[0] = StandardDeviation(y, y_center[0]);
    y_mat_scale[0] = y_scale[0];

    for (unsigned i = 0; i < n_samples; ++i) {
      y[i] -= y_center[0];
      y[i] /= y_scale[0];
      y_mat(i, 0) = y[i];
    }

    lambda_scaling = y_scale[0];
  }

  void Predict(const std::vector<double>& w,
               const double               wscale,
               const unsigned             n_features,
               const unsigned             s_ind,
               const Eigen::MatrixXd&     x) {

    double inner_product = 0.0;
    for (unsigned f_ind = 0; f_ind < n_features; ++f_ind)
      inner_product += x(f_ind, s_ind) * w[f_ind];

    prediction = wscale*inner_product + intercept;
  }

  void Predict(const std::vector<double>&         w,
               const double                       wscale,
               const unsigned                     n_features,
               const unsigned                     s_ind,
               const Eigen::SparseMatrix<double>& x) {

    double inner_product = 0.0;
    for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it)
      inner_product += it.value() * w[it.index()];

    prediction = wscale*inner_product + intercept;
  }

  double Loss(const unsigned i) {
    return 0.5*(prediction - y[i])*(prediction - y[i]);
  }

  void Gradient(const unsigned i) {
    gradient = prediction - y[i];
    gradient_change = gradient - gradient_memory[i];
    gradient_memory[i] = gradient;
  }

  void FitIntercept(const double gamma, const double intercept_decay) {
    gradient_sum_intercept += gradient_change/n_samples;
    intercept -=
      gamma*gradient_sum_intercept*intercept_decay + gradient_change/n_samples;
  }

  double NullDeviance(const bool fit_intercept) {
    prediction = Mean(y);

    double loss = 0.0;
    for (unsigned i = 0; i < y.size(); ++i)
      loss += Loss(i);

    return 2.0 * loss;
  }

  double gradient = 0.0;
  double gradient_change = 0.0;
  double gradient_sum_intercept = 0.0;
  double prediction = 0.0;
  double intercept = 0.0;
};

class Binomial : public Family {
public:
  Binomial(const std::vector<double>& y,
           const unsigned             n_samples,
           const unsigned             n_classes)
           : Family(y, n_samples, n_classes, 0.25) {}

  void PreprocessResponse() {
    auto y_mu = Mean(y);
    auto y_sd = StandardDeviation(y, y_mu);

    for (unsigned i = 0; i < n_samples; ++i)
      y_mat(i, 0) = (y[i] - y_mu)/y_sd;

    y_mat_scale[0] = y_sd;
  }

  void Predict(const std::vector<double>& w,
               const double               wscale,
               const unsigned             n_features,
               const unsigned             s_ind,
               const Eigen::MatrixXd&     x) {

    double inner_product = 0.0;
    for (unsigned f_ind = 0; f_ind < n_features; ++f_ind)
      inner_product += x(f_ind, s_ind) * w[f_ind];

    prediction = wscale*inner_product + intercept;
  }

  void Predict(const std::vector<double>&         w,
               const double                       wscale,
               const unsigned                     n_features,
               const unsigned                     s_ind,
               const Eigen::SparseMatrix<double>& x) {

    double inner_product = 0.0;
    for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it)
      inner_product += it.value() * w[it.index()];

    prediction = wscale*inner_product + intercept;
  }

  double Link(double y) {
    // TODO(jolars): let the user set this.
    double pmin = 1e-9;
    double pmax = 1.0 - pmin;
    double z = Clamp(y, pmin, pmax);

    return std::log(z / (1.0 - z));
  }

  double Loss(const unsigned i) {
    return std::log(1.0 + std::exp(prediction)) - y[i]*prediction;
  }

  void Gradient(const unsigned i) {
    gradient = 1.0 - y[i] - 1.0/(1.0 + std::exp(prediction));
    gradient_change = gradient - gradient_memory[i];
    gradient_memory[i] = gradient;
  }

  void FitIntercept(const double gamma, const double intercept_decay) {
    gradient_sum_intercept += gradient_change/n_samples;
    intercept -=
      gamma*gradient_sum_intercept*intercept_decay + gradient_change/n_samples;
  }

  double NullDeviance(const bool fit_intercept) {
    prediction = fit_intercept ? Link(Mean(y)) : 0.0;

    double loss = 0.0;
    for (unsigned i = 0; i < n_samples; ++i)
      loss += Loss(i);

    return 2.0 * loss;
  }

  double gradient = 0.0;
  double gradient_change = 0.0;
  double gradient_sum_intercept = 0.0;
  double prediction = 0.0;
  double intercept = 0.0;
};

class Multinomial : public Family {
public:
  Multinomial(const std::vector<double>& y,
              const unsigned             n_samples,
              const unsigned             n_classes)
              : Family(y, n_samples, n_classes, 0.5) {

    gradient.resize(n_classes);
    gradient_change.resize(n_classes);
    gradient_sum_intercept.resize(n_classes);
    prediction.resize(n_classes);
    intercept.resize(n_classes);
  }

  void PreprocessResponse() {
    std::vector<double> y_mu(n_classes);
    std::vector<double> y_var(n_classes);

    for (unsigned i = 0; i < n_samples; ++i) {
      unsigned y_class = std::round(y[i]);
      y_mat(i, y_class) = 1.0;
    }

    for (unsigned j = 0; j < n_classes; ++j)
      y_mu[j] = y_mat.col(j).mean();

    for (unsigned i = 0; i < n_classes; ++i) {
      for (unsigned j = 0; j < n_samples; ++j)
        y_var[i] += std::pow(y_mat(j, i) - y_mu[i], 2)/n_samples;

      y_mat_scale[i] = y_var[i] == 0.0 ? 1.0 : std::sqrt(y_var[i]);
    }

    for (unsigned i = 0; i < n_classes; ++i) {
      for (unsigned j = 0; j < n_samples; ++j) {
        y_mat(j, i) -= y_mu[i];
        y_mat(j, i) /= y_mat_scale[i];
      }
    }
  }

  void Predict(const std::vector<double>& w,
               const double               wscale,
               const unsigned             n_features,
               const unsigned             s_ind,
               const Eigen::MatrixXd&     x) {

    for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind) {
      double inner_product = 0.0;
      for (unsigned f_ind = 0; f_ind < n_features; ++f_ind)
        inner_product += x(f_ind, s_ind) * w[f_ind*n_classes + c_ind];

      prediction[c_ind] = wscale*inner_product + intercept[c_ind];
    }
  }

  void Predict(const std::vector<double>&         w,
               const double                       wscale,
               const unsigned                     n_features,
               const unsigned                     s_ind,
               const Eigen::SparseMatrix<double>& x) {

    for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind) {
      double inner_product = 0.0;
      for (Eigen::SparseMatrix<double>::InnerIterator it(x, s_ind); it; ++it)
        inner_product += it.value() * w[it.index()*n_classes + c_ind];

      prediction[c_ind] = wscale*inner_product + intercept[c_ind];
    }
  }

  double Loss(const unsigned i) {
    auto c = static_cast<unsigned>(y[i] + 0.5);
    return LogSumExp(prediction) - prediction[c];
  }

  void Gradient(const unsigned i) {

    auto lse = LogSumExp(prediction);
    auto c = static_cast<unsigned>(y[i] + 0.5);

    for (unsigned j = 0; j < n_classes; ++j) {
      gradient[j] = std::exp(prediction[j] - lse);

      if (j == c)
        gradient[j] -= 1.0;

      gradient_change[j] = gradient[j] - gradient_memory[i*n_classes + j];
      gradient_memory[i*n_classes + j] = gradient[j];
    }
  }

  void FitIntercept(const double gamma, const double intercept_decay) {
    for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind) {
      gradient_sum_intercept[c_ind] += gradient_change[c_ind]/n_samples;
      intercept[c_ind] -= gamma*gradient_sum_intercept[c_ind]*intercept_decay
        + gradient_change[c_ind]/n_samples;
    }
  }

  double NullDeviance(const bool fit_intercept) {
    if (fit_intercept)
      prediction = Proportions(y, n_classes);
    else
      prediction.assign(n_classes, 1.0/n_classes);

    double prediction_sum_avg = 0.0;

    for (auto& prediction_i : prediction) {
      prediction_i = std::log(prediction_i);
      prediction_sum_avg += prediction_i/n_classes;
    }

    for (auto& prediction_i : prediction)
      prediction_i -= prediction_sum_avg;

    double loss = 0.0;
    double lse = LogSumExp(prediction);

    for (const auto y_i : y) {
      auto c = static_cast<unsigned>(y_i + 0.5);
      loss += lse - prediction[c];
    }

    return 2.0 * loss;
  }

  std::vector<double> gradient;
  std::vector<double> gradient_change;
  std::vector<double> gradient_sum_intercept;
  std::vector<double> prediction;
  std::vector<double> intercept;
};

} // namespace sgdnet

#endif // SGDNET_FAMILIES_
