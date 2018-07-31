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
#include <RcppEigen.h>
#include "math.h"
#include "utils.h"
namespace sgdnet {

class Family {
public:
  Family(const double L_scaling) : L_scaling(L_scaling) {};

  void Preprocess(Eigen::MatrixXd&     y,
                  std::vector<double>& y_center,
                  std::vector<double>& y_scale) noexcept;

  double Loss(const std::vector<double>& prediction,
              const Eigen::MatrixXd&     y,
              const unsigned             i) const noexcept;

  void Gradient(const std::vector<double>& prediction,
                const Eigen::MatrixXd&     y,
                const unsigned             i,
                Eigen::ArrayXd&            gradient) const noexcept;

  double NullDeviance(const Eigen::MatrixXd& y,
                      const bool             fit_intercept,
                      const unsigned         n_classes) const noexcept;

  template <typename T>
  double
  LambdaMax(const T&                   x,
            const Eigen::MatrixXd&     y,
            const std::vector<double>& y_scale) const noexcept;

  double L_scaling;
};

class Gaussian : public Family {
public:
  Gaussian() : Family(1.0) {}

  void Preprocess(Eigen::MatrixXd&     y,
                  std::vector<double>& y_center,
                  std::vector<double>& y_scale) noexcept {
    y_center = Mean(y);
    y_scale = StandardDeviation(y, y_center);

    auto n = y.rows();
    for (decltype(n) i = 0; i < n; ++i)
      y(i, 0) = (y(i, 0) - y_center[0])/y_scale[0];
  }

  double Loss(const std::vector<double>& prediction,
              const Eigen::MatrixXd&     y,
              const unsigned             i) const noexcept {
    return 0.5*(prediction[0] - y(0, i))*(prediction[0] - y(0, i));
  }

  void Gradient(const std::vector<double>& prediction,
                const Eigen::MatrixXd&     y,
                const unsigned             i,
                Eigen::ArrayXd&            gradient) const noexcept {
    gradient[0] = prediction[0] - y(0, i);
  }

  double NullDeviance(const Eigen::MatrixXd& y,
                      const bool             fit_intercept,
                      const unsigned         n_classes) const noexcept {
    auto prediction = Mean(y.transpose());

    double loss = 0.0;
    auto n = y.cols();
    for (decltype(n) i = 0; i < n; ++i)
      loss += Loss(prediction, y, i);

    return 2.0 * loss;
  }

  template <typename T>
  double
  LambdaMax(const T&                   x,
            const Eigen::MatrixXd&     y,
            const std::vector<double>& y_scale) const noexcept {
    return y_scale[0]*(x.transpose() * y).cwiseAbs().maxCoeff()/x.rows();
  }
};

class Binomial : public Family {
public:
  Binomial() : Family(0.25) {}

  void Preprocess(Eigen::MatrixXd&     y,
                  std::vector<double>& y_center,
                  std::vector<double>& y_scale) const noexcept {
    // no preprocessing
  }

  double Link(double y) const noexcept {
    // TODO(jolars): let the user set this.
    double pmin = 1e-9;
    double pmax = 1.0 - pmin;
    double z = Clamp(y, pmin, pmax);

    return std::log(z / (1.0 - z));
  }

  double Loss(const std::vector<double>& prediction,
              const Eigen::MatrixXd&     y,
              const unsigned             i) const noexcept {
    return std::log(1.0 + std::exp(prediction[0])) - y(0, i)*prediction[0];
  }

  void Gradient(const std::vector<double>& prediction,
                const Eigen::MatrixXd&     y,
                const unsigned             i,
                Eigen::ArrayXd&            gradient) const noexcept {
    gradient[0] = 1.0 - y(0, i) - 1.0/(1.0 + std::exp(prediction[0]));
  }

  double NullDeviance(const Eigen::MatrixXd& y,
                      const bool             fit_intercept,
                      const unsigned         n_classes) const noexcept {

    std::vector<double> prediction(1);

    if (fit_intercept) {
      auto y_bar = Mean(y.transpose());
      prediction[0] = Link(y_bar[0]);
    }

    double loss = 0.0;
    for (unsigned i = 0; i < y.cols(); ++i)
      loss += Loss(prediction, y, i);

    return 2.0 * loss;
  }

  template <typename T>
  double
  LambdaMax(const T&                   x,
            const Eigen::MatrixXd&     y,
            const std::vector<double>& y_scale) const noexcept {

    auto n = y.rows();

    Eigen::VectorXd y_map(n);

    auto y_bar = Mean(y);
    auto y_std = StandardDeviation(y, y_bar);

    for (decltype(n) i = 0; i < n; ++i)
      y_map(i) = (y(i, 0) - y_bar[0])/y_std[0];

    return y_std[0]*(x.transpose() * y_map).cwiseAbs().maxCoeff()/n;
  }
};

class Multinomial : public Family {
public:
  Multinomial(const unsigned n_classes) : Family(0.25), n_classes(n_classes) {}

  void Preprocess(Eigen::MatrixXd&     y,
                  std::vector<double>& y_center,
                  std::vector<double>& y_scale) const noexcept {
    // no preprocessing
  }

  double Loss(const std::vector<double>& prediction,
              const Eigen::MatrixXd&     y,
              const unsigned             i) const noexcept {
    auto c = static_cast<unsigned>(y(0, i) + 0.5);
    return LogSumExp(prediction) - prediction[c];
  }

  void Gradient(const std::vector<double>& prediction,
                const Eigen::MatrixXd&     y,
                const unsigned             i,
                Eigen::ArrayXd&            gradient) const noexcept {

    auto lse = LogSumExp(prediction);
    unsigned p = prediction.size();
    auto c = static_cast<unsigned>(y(0, i) + 0.5);

    for (decltype(p) j = 0; j < p; ++j) {
      gradient[j] = std::exp(prediction[j] - lse);

      if (j == c)
        gradient[j] -= 1.0;
    }
  }

  double NullDeviance(const Eigen::MatrixXd& y,
                      const bool             fit_intercept,
                      const unsigned         n_classes) const noexcept {
    std::vector<double> prediction;

    auto n_samples = y.cols();

    if (fit_intercept)
      prediction = Proportions(y, n_classes);
    else
      prediction.resize(n_classes, 1.0/n_classes);

    auto prediction_sum_avg = 0.0;

    for (auto& prediction_i : prediction) {
      prediction_i = std::log(prediction_i);
      prediction_sum_avg += prediction_i/n_classes;
    }

    for (auto& prediction_i : prediction)
      prediction_i -= prediction_sum_avg;

    auto loss = 0.0;
    auto lse = LogSumExp(prediction);

    for (decltype(n_samples) i = 0; i < n_samples; ++i) {
      auto c = static_cast<unsigned>(y(0, i) + 0.5);
      loss += lse - prediction[c];
    }

    return 2.0 * loss;
  }

  template <typename T>
  double
  LambdaMax(const T&                   x,
            const Eigen::MatrixXd&     y,
            const std::vector<double>& y_scale) const noexcept {
    auto n = y.rows();

    Eigen::MatrixXd y_map = Eigen::MatrixXd::Zero(n, n_classes);

    for (decltype(n) i = 0; i < n; ++i) {
      auto c = static_cast<unsigned>(y(i, 0) + 0.5);
      y_map(i, c) = 1.0;
    }

    auto y_bar = Mean(y_map);
    auto y_std = StandardDeviation(y_map, y_bar);
    Standardize(y_map, y_bar, y_std);

    Eigen::MatrixXd inner_products = y_map.transpose() * x;

    double max_coeff = 0.0;
    auto m = inner_products.cols();
    auto p = inner_products.rows();

    for (decltype(m) j = 0; j < m; ++j) {
      for (decltype(p) k = 0; k < p; ++k) {
        max_coeff = std::max(std::abs(inner_products(k, j)*y_std[k]),
                             max_coeff);
      }
    }

    return max_coeff/n;
  }

private:
  const unsigned n_classes;
};

class MultivariateGaussian : public Family {
public:
  MultivariateGaussian(const unsigned n_classes,
                       const bool standardize_response)
                       : Family(1.0),
                         n_classes(n_classes),
                         standardize_response(standardize_response) {}

  void Preprocess(Eigen::MatrixXd&     y,
                  std::vector<double>& y_center,
                  std::vector<double>& y_scale) const noexcept {
    // TODO(jolars): setup preprocessing for multivariate family, and
    // condition on user input
    if (standardize_response) {
      y_center = Mean(y);
      y_scale = StandardDeviation(y, y_center);

      Standardize(y, y_center, y_scale);
    }
  }

  double Loss(const std::vector<double>& prediction,
              const Eigen::MatrixXd&     y,
              const unsigned             i) const noexcept {

    double loss = 0.0;
    for (unsigned k = 0; k < n_classes; ++k)
      loss += 0.5*std::pow(prediction[k] - y(k, i), 2);

    return loss;
  }

  void Gradient(const std::vector<double>& prediction,
                const Eigen::MatrixXd&     y,
                const unsigned             i,
                Eigen::ArrayXd&            gradient) const noexcept {

    for (unsigned k = 0; k < n_classes; ++k)
      gradient[k] = prediction[k] - y(k, i);
  }

  double NullDeviance(const Eigen::MatrixXd& y,
                      const bool             fit_intercept,
                      const unsigned         n_classes) const noexcept {

    auto conditional_mean = Mean(y.transpose());

    double loss = 0.0;
    for (decltype(y.cols()) i = 0; i < y.cols(); ++i)
      loss += Loss(conditional_mean, y, i);

    return 2.0 * loss;
  }

  template <typename T>
  double
  LambdaMax(const T&                   x,
            const Eigen::MatrixXd&     y,
            const std::vector<double>& y_scale) const noexcept {
    Eigen::MatrixXd y_map(y);

    auto y_bar = Mean(y);
    auto y_std = StandardDeviation(y, y_bar);

    Standardize(y_map, y_bar, y_std);

    Eigen::ArrayXXd inner_products = x.transpose() * y_map;

    for (unsigned k = 0; k < n_classes; ++k)
      inner_products.col(k) *= y_scale[k]*y_std[k];

    return inner_products.square().rowwise().sum().sqrt().maxCoeff()/y.rows();
  }

private:
  const unsigned n_classes;
  const bool standardize_response;
};

} // namespace sgdnet

#endif // SGDNET_FAMILIES_
