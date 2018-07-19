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
         const unsigned n_samples,
         const unsigned n_classes,
         const double L_scaling)
         : y(y),
           n_samples(n_samples),
           n_classes(n_classes),
           L_scaling(L_scaling) {
    // Initialize scaling and centering parameters for the response
    y_center.resize(n_classes, 0.0);
    y_scale.resize(n_classes, 1.0);
    y_mat.setZero(n_samples, n_classes);
    y_mat_scale.resize(n_classes, 1.0);
  };

  virtual double Link(const unsigned i) = 0;

  virtual double Loss(const double prediction, const unsigned i) = 0;

  virtual double Loss(const std::vector<double>& prediction,
                      const unsigned i) = 0;

  virtual void Gradient(std::vector<double>&       g_i,
                        const std::vector<double>& prediction,
                        const unsigned             i) = 0;

  virtual void PreprocessResponse() = 0;

  virtual double NullDeviance(const bool fit_intercept) = 0;

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
           const unsigned n_samples,
           const unsigned n_classes)
           : Family(y, n_samples, n_classes, 1.0) {}

  void PreprocessResponse() {
    y_center[0] = Mean(y);
    y_scale[0] = StandardDeviation(y, y_center[0]);
    y_mat_scale[0] = y_scale[0];

    for (std::size_t i = 0; i < n_samples; ++i) {
      y[i] -= y_center[0];
      y[i] /= y_scale[0];
      y_mat(i, 0) = y[i];
    }

    lambda_scaling = y_scale[0];
  }

  double Link(const unsigned i) { return y[i]; }

  double Loss(const double prediction, const unsigned i) {
    return 0.5*(prediction - y[i])*(prediction - y[i]);
  }

  double Loss(const std::vector<double>& prediction, const unsigned i) {
    return Loss(prediction[0], i);
  }

  void Gradient(std::vector<double>&       g_i,
                const std::vector<double>& prediction,
                const unsigned             i) {
    g_i[0] = prediction[0] - y[i];
  }

  double NullDeviance(const bool fit_intercept) {
    auto y_mu = Mean(y);

    double loss = 0.0;
    for (unsigned i = 0; i < y.size(); ++i)
      loss += Loss(y_mu, i);

    return 2.0 * loss;
  }
};

class Binomial : public Family {
public:
  Binomial(const std::vector<double>& y,
           const unsigned n_samples,
           const unsigned n_classes)
           : Family(y, n_samples, n_classes, 0.25) {}

  void PreprocessResponse() {
    auto y_mu = Mean(y);
    auto y_sd = StandardDeviation(y, y_mu);

    for (std::size_t i = 0; i < n_samples; ++i)
      y_mat(i, 0) = (y[i] - y_mu)/y_sd;

    y_mat_scale[0] = y_sd;
  }

  double Link(double y) {
    // TODO(jolars): let the user set this.
    double pmin = 1e-9;
    double pmax = 1.0 - pmin;
    double z = Clamp(y, pmin, pmax);

    return std::log(z / (1.0 - z));
  }

  double Link(const unsigned i) {
    return Link(y[i]);
  }

  double Loss(const double prediction, const unsigned i) {
    return std::log(1.0 + std::exp(prediction)) - y[i]*prediction;
  }

  double Loss(const std::vector<double>& prediction, const unsigned i) {
    return Loss(prediction[0], i);
  }

  void Gradient(std::vector<double>&       g_i,
                const std::vector<double>& prediction,
                const unsigned             i) {
    g_i[0] = 1.0 - y[i] - 1.0/(1.0 + std::exp(prediction[0]));
  }

  double NullDeviance(const bool fit_intercept) {
    double pred = fit_intercept ? Link(Mean(y)) : 0.0;

    double loss = 0.0;
    for (unsigned i = 0; i < n_samples; ++i)
      loss += Loss(pred, i);

    return 2.0 * loss;
  }
};

class Multinomial : public Family {
public:
  Multinomial(const std::vector<double>& y,
              const unsigned n_samples,
              const unsigned n_classes)
              : Family(y, n_samples, n_classes, 0.5) {}

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

  double Link(const unsigned i) {
    return std::log(y[i] / (1.0 - y[i]));
  }

  double Loss(const double prediction, const unsigned i) {
    // Not reasonable for multinomial model
    return 0.0;
  }

  double Loss(const std::vector<double>& prediction, const unsigned i) {

    unsigned y_class = std::round(y[i]);

    return LogSumExp(prediction) - prediction[y_class];
  }

  void Gradient(std::vector<double>&       g_i,
                const std::vector<double>& prediction,
                const unsigned             i) {

    auto lse = LogSumExp(prediction);
    unsigned y_i = static_cast<unsigned>(y[i] + 0.5);

    for (unsigned c_ind = 0; c_ind < n_classes; ++c_ind) {
      g_i[c_ind] = std::exp(prediction[c_ind] - lse);

      if (c_ind == y_i)
        g_i[c_ind] -= 1.0;
    }
  }

  double NullDeviance(const bool fit_intercept) {
    std::vector<double> pred;

    if (fit_intercept)
      pred = Proportions(y, n_classes);
    else
      pred.resize(n_classes, 1.0/n_classes);

    double pred_sum_avg = 0.0;

    for (auto& pred_i : pred) {
      pred_i = std::log(pred_i);
      pred_sum_avg += pred_i/n_classes;
    }

    for (auto& pred_i : pred)
      pred_i -= pred_sum_avg;

    double loss = 0.0;
    double lse = LogSumExp(pred);

    for (unsigned i = 0; i < n_samples; ++i) {
      unsigned y_class = static_cast<unsigned>(y[i] + 0.5);
      loss += lse - pred[y_class];
    }

    return 2.0 * loss;
  }
};

} // namespace sgdnet

#endif // SGDNET_FAMILIES_
