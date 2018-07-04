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

#include <RcppArmadillo.h>
#include <memory>
#include "math.h"

namespace sgdnet {

class Family {
public:
  virtual ~Family() {};

  virtual double Loss(const double prediction, const double y) = 0;

  virtual double Gradient(const double prediction, const double y) = 0;

  virtual void PreprocessResponse(std::vector<double>& y,
                                  std::vector<double>& y_center,
                                  std::vector<double>& y_scale,
                                  const bool           fit_intercept) = 0;

  virtual double NullDeviance(const std::vector<double>& y) = 0;

  std::vector<double> StepSize(const double               max_squared_sum,
                               const std::vector<double>& alpha_scaled,
                               const bool                 fit_intercept,
                               const std::size_t          n_samples) {
    // Lipschitz constant approximation
    std::vector<double> step_sizes;
    step_sizes.reserve(alpha_scaled.size());

    for (auto alpha_val : alpha_scaled) {
      double L = L_scaling*(max_squared_sum + fit_intercept) + alpha_val;
      double mu_n = 2.0*n_samples*alpha_val;
      step_sizes.push_back(1.0 / (2.0*L + std::min(L, mu_n)));
    }
    return step_sizes;
  };

  std::size_t GetNClasses() {
    return n_classes;
  }

protected:
  std::size_t n_classes;
  double L_scaling;
};

class Gaussian : public Family {
public:
  Gaussian() {
    n_classes = 1;
    L_scaling = 1.0;
  };

  virtual ~Gaussian() {};

  double Loss(const double prediction, const double y) {
    return 0.5*(prediction - y)*(prediction - y);
  };

  double Gradient(const double prediction, const double y) {
    return prediction - y;
  };

  void PreprocessResponse(std::vector<double>& y,
                          std::vector<double>& y_center,
                          std::vector<double>& y_scale,
                          const bool           fit_intercept) {
    if (fit_intercept) {
      double y_mu = Mean(y, y.size());
      double y_sd = StandardDeviation(y, y.size());

      y_center.push_back(y_mu);
      y_scale.push_back(y_sd);

      for (auto& y_val : y) {
        y_val -= y_mu;

        if (y_sd != 0)
          y_val /= y_sd;
      }
    } else {
      y_center.push_back(0.0);
      y_scale.push_back(1.0);
    }
  };

  double NullDeviance(const std::vector<double>& y) {
    double y_mu = Mean(y, y.size());
    double loss = 0.0;

    for (const auto& y_val : y)
      loss += Loss(y_mu, y_val);

    return 2.0 * loss;
  }
};

class Binomial : public Family {
public:
  Binomial() {
    n_classes = 1;
    L_scaling = 0.25;
  };

  virtual ~Binomial() {};

  double Loss(const double prediction, const double y) {
    return std::log(1.0 + std::exp(prediction)) - y*prediction;
  };

  double Gradient(const double prediction, const double y) {
    return 1.0 - y - 1.0/(1.0 + std::exp(prediction));
  };

  void PreprocessResponse(std::vector<double>& y,
                          std::vector<double>& y_center,
                          std::vector<double>& y_scale,
                          const bool           fit_intercept) {
    y_center.push_back(0.0);
    y_scale.push_back(1.0);
  };

  double NullDeviance(const std::vector<double>& y) {
    double y_mu = Mean(y, y.size());
    double y_mu_log = std::log(y_mu / (1.0 - y_mu));
    double loss = 0.0;

    for (auto const& y_val : y)
      loss += Loss(y_mu_log, y_val);

    return 2.0 * loss;
  }
};

class FamilyFactory {
public:
  static std::unique_ptr<Family> NewFamily(const std::string& family_choice) {
    if (family_choice == "gaussian")
      return std::unique_ptr<Family>(new Gaussian());
    else if (family_choice == "binomial")
      return std::unique_ptr<Family>(new Binomial());
    return NULL;
  };
};

} // namespace sgdnet

#endif // SGDNET_FAMILIES_
