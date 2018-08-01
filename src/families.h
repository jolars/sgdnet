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

  void Preprocess(std::vector<double>& y,
                  std::vector<double>& y_center,
                  std::vector<double>& y_scale) noexcept;

  double Loss(const std::vector<double>& prediction,
              const std::vector<double>& y,
              const unsigned             i) const noexcept;

  void Gradient(const std::vector<double>& prediction,
                const std::vector<double>& y,
                const unsigned             i,
                std::vector<double>&       gradient) const noexcept;

  double NullDeviance(const std::vector<double>& y,
                      const bool                 fit_intercept,
                      const unsigned             n_classes) const noexcept;

  Eigen::MatrixXd
  LambdaResponse(const std::vector<double>& y,
                 std::vector<double>&       y_mat_scale) const noexcept;

  double L_scaling;
};

class Gaussian : public Family {
public:
  Gaussian() : Family(1.0) {}

  void Preprocess(std::vector<double>& y,
                  std::vector<double>& y_center,
                  std::vector<double>& y_scale) noexcept {
    y_center[0] = Mean(y);
    y_scale[0] = StandardDeviation(y, y_center[0]);

    for (auto& y_i : y)
      y_i = (y_i - y_center[0])/y_scale[0];
  }

  double Loss(const std::vector<double>& prediction,
              const std::vector<double>& y,
              const unsigned             i) const noexcept {
    return 0.5*(prediction[0] - y[i])*(prediction[0] - y[i]);
  }

  void Gradient(const std::vector<double>& prediction,
                const std::vector<double>& y,
                const unsigned             i,
                std::vector<double>&       gradient) const noexcept {
    gradient[0] = prediction[0] - y[i];
  }

  double NullDeviance(const std::vector<double>& y,
                      const bool                 fit_intercept,
                      const unsigned             n_classes) const noexcept {
    std::vector<double> prediction(1, Mean(y));

    double loss = 0.0;
    for (unsigned i = 0; i < y.size(); ++i)
      loss += Loss(prediction, y, i);

    return 2.0 * loss;
  }

  Eigen::MatrixXd
  LambdaResponse(const std::vector<double>& y,
                 std::vector<double>&       y_mat_scale) const noexcept {
    Eigen::MatrixXd y_mat(y.size(), 1);

    auto y_mu = Mean(y);
    auto y_sd = StandardDeviation(y, y_mu);

    for (unsigned i = 0; i < y.size(); ++i)
      y_mat(i, 0) = (y[i] - y_mu)/y_sd;

    y_mat_scale[0] = y_sd;

    return y_mat;
  }
};

class Binomial : public Family {
public:
  Binomial() : Family(0.25) {}

  void Preprocess(std::vector<double>& y,
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
              const std::vector<double>& y,
              const unsigned             i) const noexcept {
    return std::log(1.0 + std::exp(prediction[0])) - y[i]*prediction[0];
  }

  void Gradient(const std::vector<double>& prediction,
                const std::vector<double>& y,
                const unsigned             i,
                std::vector<double>&       gradient) const noexcept {
    gradient[0] = 1.0 - y[i] - 1.0/(1.0 + std::exp(prediction[0]));
  }

  double NullDeviance(const std::vector<double>& y,
                      const bool                 fit_intercept,
                      const unsigned             n_classes) const noexcept {
    double y_mu = fit_intercept ? Link(Mean(y)) : 0.0;
    std::vector<double> prediction(1, y_mu);

    double loss = 0.0;
    for (unsigned i = 0; i < y.size(); ++i)
      loss += Loss(prediction, y, i);

    return 2.0 * loss;
  }

  Eigen::MatrixXd
  LambdaResponse(const std::vector<double>& y,
                 std::vector<double>&       y_mat_scale) const noexcept {

    Eigen::MatrixXd y_mat(y.size(), 1);

    auto y_mu = Mean(y);
    auto y_sd = StandardDeviation(y, y_mu);

    for (unsigned i = 0; i < y.size(); ++i)
      y_mat(i, 0) = (y[i] - y_mu)/y_sd;

    y_mat_scale[0] = y_sd;

    return y_mat;
  }
};

class Multinomial : public Family {
public:
  Multinomial() : Family(0.25) {}

  void Preprocess(std::vector<double>& y,
                  std::vector<double>& y_center,
                  std::vector<double>& y_scale) const noexcept {
    // no preprocessing
  }

  double Loss(const std::vector<double>& prediction,
              const std::vector<double>& y,
              const unsigned             i) const noexcept {
    auto c = static_cast<unsigned>(y[i] + 0.5);
    return LogSumExp(prediction) - prediction[c];
  }

  void Gradient(const std::vector<double>& prediction,
                const std::vector<double>& y,
                const unsigned             i,
                std::vector<double>&       gradient) const noexcept {

    auto lse = LogSumExp(prediction);
    auto c = static_cast<unsigned>(y[i] + 0.5);

    for (unsigned j = 0; j < prediction.size(); ++j) {
      gradient[j] = std::exp(prediction[j] - lse);

      if (j == c)
        gradient[j] -= 1.0;
    }
  }

  double NullDeviance(const std::vector<double>& y,
                      const bool                 fit_intercept,
                      const unsigned             n_classes) const noexcept {
    std::vector<double> prediction;

    if (fit_intercept)
      prediction = Proportions(y, n_classes);
    else
      prediction.resize(n_classes, 1.0/n_classes);

    double prediction_sum_avg = 0.0;

    for (auto& prediction_i : prediction) {
      prediction_i = std::log(prediction_i);
      prediction_sum_avg += prediction_i/n_classes;
    }

    for (auto& prediction_i : prediction)
      prediction_i -= prediction_sum_avg;

    double loss = 0.0;
    double lse = LogSumExp(prediction);

    for (auto y_i : y) {
      auto c = static_cast<unsigned>(y_i + 0.5);
      loss += lse - prediction[c];
    }

    return 2.0 * loss;
  }

  Eigen::MatrixXd
  LambdaResponse(const std::vector<double>& y,
                 std::vector<double>&       y_mat_scale) const noexcept {
    auto n_samples = y.size();
    auto n_classes = y_mat_scale.size();

    Eigen::MatrixXd y_mat = Eigen::MatrixXd::Zero(n_samples, n_classes);

    std::vector<double> y_mu(n_classes);
    std::vector<double> y_var(n_classes);

    for (unsigned i = 0; i < n_samples; ++i) {
      unsigned c = static_cast<unsigned>(y[i] + 0.5);
      y_mat(i, c) = 1.0;
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

    return y_mat;
  }
};

} // namespace sgdnet

#endif // SGDNET_FAMILIES_
