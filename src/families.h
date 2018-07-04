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

namespace sgdnet {

class Family {
public:
  virtual double Link(const double y) = 0;

  virtual double Loss(const double prediction, const double y) = 0;

  virtual double Gradient(const double prediction, const double y) = 0;

  virtual void PreprocessResponse(std::vector<double>& y,
                                  std::vector<double>& y_center,
                                  std::vector<double>& y_scale) = 0;

  virtual double NullDeviance(const std::vector<double>& y) = 0;

  virtual void Prox(const double prediction,
                    const double y,
                    const double gamma_scaled,
                    double& xp,
                    double& sg) = 0;

  double ProxNewtonStep(double y,
                        double x2,
                        double gamma_scaled,
                        double ry) {
    double expy = std::exp(y*ry);
    double sigma = -ry/(1 + expy);
    double gamma_scaled_sigma = gamma_scaled*sigma;
    double numerator = gamma_scaled_sigma  + (y-x2);
    double denominator = 1 - ry*gamma_scaled_sigma- gamma_scaled_sigma*sigma;

    return(numerator/denominator);
  }

  std::vector<double> StepSize(const double               max_squared_sum,
                               const std::vector<double>& alpha_scaled,
                               const bool                 fit_intercept,
                               const std::size_t          n_samples) {
    // Lipschitz constant approximation
    std::vector<double> step_sizes;
    step_sizes.reserve(alpha_scaled.size());

    for (auto alpha_val : alpha_scaled) {
      double L =
        L_scaling_*(max_squared_sum + static_cast<double>(fit_intercept))
        + alpha_val;
      double mu_n = 2.0*n_samples*alpha_val;
      step_sizes.push_back(1.0 / (2.0*L + std::min(L, mu_n)));
    }
    return step_sizes;
  }

  unsigned n_classes() { return n_classes_; }
  double lambda_scaling() { return lambda_scaling_; }

protected:
  unsigned n_classes_;
  double L_scaling_;
  double lambda_scaling_;
};

class Gaussian : public Family {
public:
  Gaussian() {
    n_classes_ = 1;
    L_scaling_ = 1.0;
    lambda_scaling_ = 1.0;
  }

  double Link(const double y) { return y; }

  double Loss(const double prediction, const double y) {
    return 0.5*(prediction - y)*(prediction - y);
  }

  double Gradient(const double prediction, const double y) {
    return prediction - y;
  }

  void PreprocessResponse(std::vector<double>& y,
                          std::vector<double>& y_center,
                          std::vector<double>& y_scale) {
    double y_mu = Mean(y);
    double y_sd = StandardDeviation(y);

    if (y_sd == 0.0) y_sd = 1.0;

    y_center[0] = y_mu;
    y_scale[0] = y_sd;

    for (auto& y_val : y) {
      y_val -= y_mu;
      y_val /= y_sd;
    }
  }

  double NullDeviance(const std::vector<double>& y) {
    double y_mu = Mean(y);

    double loss = 0.0;
    for (const auto y_i : y)
      loss += Loss(y_mu, y_i);

    return 2.0 * loss;
  }

  void Prox(const double prediction,
            const double y,
            const double gamma_scaled,
            double& xp,
            double& sg) {
  }
};

class Binomial : public Family {
public:
  Binomial() {
    n_classes_ = 1;
    L_scaling_ = 0.25;
    lambda_scaling_ = 0.5;
  }

  double Link(const double y) {
    return std::log(y / (1.0 - y));
  }

  double Loss(const double prediction, const double y) {
    double z = prediction * y;

    if (z > 18.0)
      return std::exp(-z);
    if (z < -18.0)
      return -z;

    return std::log(1.0 + std::exp(-z));
  }

  double Gradient(const double prediction, const double y) {
    double z = prediction * y;

    if (z > 18.0)
      return std::exp(-z) * -y;
    if (z < -18.0)
      return -y;

    return -y / (std::exp(z) + 1.0);
  }

  void PreprocessResponse(std::vector<double>& y,
                          std::vector<double>& y_center,
                          std::vector<double>& y_scale) {
    // No preprocessing for the response in the binomial case
  }

  double NullDeviance(const std::vector<double>& y) {
    double y_mu = Mean(y)/2.0 + 0.5;

    double loss = 0.0;
    for (const auto y_i : y)
      loss += Loss(Link(y_mu), Link(y_i));

    return 2.0 * loss;
  }

  void Prox(const double prediction,
            const double y,
            const double gamma_scaled,
            double& xp,
            double& sg) {

    for (int i = 0; i < 12; ++i)
      xp -= ProxNewtonStep(xp, prediction, gamma_scaled, y);

    sg = (prediction - xp)/gamma_scaled;
  }
};

} // namespace sgdnet

#endif // SGDNET_FAMILIES_
