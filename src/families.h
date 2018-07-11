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
  };

  virtual double Link(const unsigned i) = 0;

  virtual double Loss(const double prediction, const unsigned i) = 0;

  virtual double Loss(const std::vector<double>& prediction,
                      const unsigned i) = 0;

  virtual void Gradient(std::vector<double>&       g_i,
                        const std::vector<double>& prediction,
                        const unsigned             i) = 0;

  virtual void PreprocessResponse() = 0;

  virtual double NullDeviance() = 0;

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

  virtual double LambdaMax(const Eigen::MatrixXd& x) = 0;
  virtual double LambdaMax(const Eigen::SparseMatrix<double>& x) = 0;

  std::vector<double> y;
  std::vector<double> y_center;
  std::vector<double> y_scale;
  unsigned n_samples;
  unsigned n_classes;
  unsigned n_targets;
  double L_scaling;
  double lambda_scaling = 1.0;
};

class Gaussian : public Family {
public:
  Gaussian(const std::vector<double>& y,
           const unsigned n_samples,
           const unsigned n_classes)
           : Family(y, n_samples, n_classes, 1.0) {}

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

  void PreprocessResponse() {
    double y_mu = Mean(y);
    double y_sd = StandardDeviation(y, y_mu);

    y_center[0] = y_mu;
    y_scale[0] = y_sd;

    for (auto& y_i : y) {
      y_i -= y_mu;
      y_i /= y_sd;
    }

    lambda_scaling = y_sd;
  }

  double NullDeviance() {
    auto y_mu = Mean(y);

    double loss = 0.0;
    for (unsigned i = 0; i < y.size(); ++i)
      loss += Loss(y_mu, i);

    return 2.0 * loss;
  }

  double LambdaMax(const Eigen::MatrixXd& x) {
    Eigen::RowVectorXd yt = Eigen::RowVectorXd::Map(y.data(), y.size());
    Eigen::VectorXd res = yt * x.transpose();

    return lambda_scaling*res.cwiseAbs().maxCoeff()/n_samples;
  }

  double LambdaMax(const Eigen::SparseMatrix<double>& x) {
    Eigen::RowVectorXd yt = Eigen::RowVectorXd::Map(y.data(), y.size());
    Eigen::VectorXd res = yt * x.transpose();

    return lambda_scaling*res.cwiseAbs().maxCoeff()/n_samples;
  }
};

class Binomial : public Family {
public:
  Binomial(const std::vector<double>& y,
           const unsigned n_samples,
           const unsigned n_classes)
           : Family(y, n_samples, n_classes, 0.25) {}

  double Link(const double y) {
    return std::log(y / (1.0 - y));
  }

  double Link(const unsigned i) {
    return Link(y[i]);
  }

  double Loss(const double prediction, const unsigned i) {
    double z = prediction * y[i];

    if (z > 18.0)
      return std::exp(-z);
    if (z < -18.0)
      return -z;

    return std::log(1.0 + std::exp(-z));
  }

  double Loss(const std::vector<double>& prediction, const unsigned i) {
    return Loss(prediction[0], i);
  }

  void Gradient(std::vector<double>&       g_i,
                const std::vector<double>& prediction,
                const unsigned             i) {
    double z = prediction[0] * y[i];

    if (z > 18.0)
      g_i[0] = std::exp(-z) * -y[i];
    else if (z < -18.0)
      g_i[0] = -y[i];
    else
      g_i[0] = -y[i] / (std::exp(z) + 1.0);
  }

  void PreprocessResponse() {
    // Do not preprocess binomial outcomes
  }

  double NullDeviance() {
    auto y_mu = Mean(y)/2.0 + 0.5;

    double loss = 0.0;
    for (unsigned i = 0; i < y.size(); ++i)
      loss += Loss(Link(y_mu), Link(i));

    return 2.0 * loss;
  }

  double LambdaMax(const Eigen::MatrixXd& x) {
    std::vector<double> z(y);

    for (auto& z_i : z)
      z_i = std::max(z_i, 0.0);

    auto z_mu = Mean(z);
    auto z_sd = StandardDeviation(z, z_mu);

    for (auto& z_i : z) {
      z_i -= z_mu;
      z_i /= z_sd;
    }

    Eigen::RowVectorXd zt = Eigen::RowVectorXd::Map(z.data(), z.size());
    Eigen::VectorXd res = zt * x.transpose();

    return res.cwiseAbs().maxCoeff()*z_sd/n_samples;
  }

  double LambdaMax(const Eigen::SparseMatrix<double>& x) {
    std::vector<double> z(y);

    for (auto& z_i : z)
      z_i = std::max(z_i, 0.0);

    auto z_mu = Mean(z);
    auto z_sd = StandardDeviation(z, z_mu);

    for (auto& z_i : z) {
      z_i -= z_mu;
      z_i /= z_sd;
    }

    Eigen::RowVectorXd zt = Eigen::RowVectorXd::Map(z.data(), z.size());
    Eigen::VectorXd res = zt * x.transpose();

    return res.cwiseAbs().maxCoeff()*z_sd/n_samples;
  }
};

class Multinomial : public Family {
public:
  Multinomial(const std::vector<double>& y,
              const unsigned n_samples,
              const unsigned n_classes)
              : Family(y, n_samples, n_classes, 0.5) {}

  double Link(const unsigned i) {
    return std::log(y[i] / (1.0 - y[i]));
  }

  double Loss(const double prediction, const unsigned i) {
    // Not reasonable for multinomial
    return 0.0;
  }

  double Loss(const std::vector<double>& prediction, const unsigned i) {

    unsigned y_i = static_cast<unsigned>(y[i] + 0.5);

    return LogSumExp(prediction) - prediction[y_i];
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

  void PreprocessResponse() {}

  double NullDeviance() {
    std::vector<double> pred(n_classes);

    // null prediction using only proportions
    for (auto y_i : y)
      pred[static_cast<unsigned>(y_i + 0.5)] += 1.0/n_samples;

    double lse_pred = LogSumExp(pred);

    double loss = 0.0;
    for (auto y_i : y)
      loss += lse_pred - pred[static_cast<unsigned>(y_i + 0.5)];

    return 2.0 * loss;
  }

  double LambdaMax(const Eigen::MatrixXd& x) {
    Eigen::MatrixXd y_tmp = Eigen::MatrixXd::Zero(n_samples, n_classes);
    std::vector<double> y_scale_(n_classes);
    std::vector<double> y_center_(n_classes);
    std::vector<double> y_var_(n_classes);

    for (unsigned i = 0; i < n_samples; ++i) {
      unsigned y_i = static_cast<unsigned>(y[i] + 0.5);
      y_tmp(i, y_i) = 1.0;
    }

    for (unsigned j = 0; j < n_classes; ++j)
      y_center_[j] = y_tmp.col(j).mean();

    for (unsigned i = 0; i < n_classes; ++i) {
      for (unsigned j = 0; j < n_samples; ++j)
        y_var_[i] += std::pow(y_tmp(j, i) - y_center_[i], 2)/n_samples;

      y_scale_[i] = y_var_[i] == 0.0 ? 1.0 : std::sqrt(y_var_[i]);
    }

    for (unsigned i = 0; i < n_classes; ++i) {
      for (unsigned j = 0; j < n_samples; ++j) {
        y_tmp(j, i) -= y_center_[i];
        y_tmp(j, i) /= y_scale_[i];
      }
    }

    Eigen::MatrixXd res = y_tmp.transpose() * x.transpose();

    double max_coeff = 0.0;

    for (unsigned i = 0; i < res.cols(); ++i) {
      for (unsigned j = 0; j < res.rows(); ++j) {
        max_coeff = std::max(std::abs(res(j, i)*y_scale_[j]), max_coeff);
      }
    }

    return max_coeff/n_samples;
  }

  double LambdaMax(const Eigen::SparseMatrix<double>& x) {
    Eigen::MatrixXd y_tmp = Eigen::MatrixXd::Zero(n_samples, n_classes);
    std::vector<double> y_scale_(n_classes);
    std::vector<double> y_center_(n_classes);
    std::vector<double> y_var_(n_classes);

    for (unsigned i = 0; i < n_samples; ++i) {
      unsigned y_i = static_cast<unsigned>(y[i] + 0.5);
      y_tmp(i, y_i) = 1.0;
    }

    for (unsigned j = 0; j < n_classes; ++j)
      y_center_[j] = y_tmp.col(j).mean();

    for (unsigned i = 0; i < n_classes; ++i) {
      for (unsigned j = 0; j < n_samples; ++j)
        y_var_[i] += std::pow(y_tmp(j, i) - y_center_[i], 2)/n_samples;

      y_scale_[i] = y_var_[i] == 0.0 ? 1.0 : std::sqrt(y_var_[i]);
    }

    for (unsigned i = 0; i < n_classes; ++i) {
      for (unsigned j = 0; j < n_samples; ++j) {
        y_tmp(j, i) -= y_center_[i];
        y_tmp(j, i) /= y_scale_[i];
      }
    }

    Eigen::MatrixXd res = y_tmp.transpose() * x.transpose();

    double max_coeff = 0.0;

    for (unsigned i = 0; i < res.cols(); ++i) {
      for (unsigned j = 0; j < res.rows(); ++j) {
        max_coeff = std::max(std::abs(res(j, i)*y_scale_[j]), max_coeff);
      }
    }

    return max_coeff/n_samples;
  }
};

} // namespace sgdnet

#endif // SGDNET_FAMILIES_
