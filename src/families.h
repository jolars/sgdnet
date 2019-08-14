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

  void
  Preprocess(Eigen::MatrixXd& y, // samples in rows
             Eigen::ArrayXd&  y_center,
             Eigen::ArrayXd&  y_scale) noexcept;

  double
  Loss(const Eigen::ArrayXd&  linear_predictor,
       const Eigen::MatrixXd& y, // samples in columns
       const unsigned         i) const noexcept;

  void
  Gradient(const Eigen::ArrayXd&  linear_predictor,
           const Eigen::MatrixXd& y, // samples in columns
           const unsigned         i,
           Eigen::ArrayXd&        gradient) const noexcept;

  double
  NullDeviance(const Eigen::MatrixXd& y, // samples in rows
               const bool             fit_intercept) const noexcept;

  void
  FitNullModel(const Eigen::MatrixXd& y, // samples in rows
               const bool             fit_intercept,
               Eigen::ArrayXd&        intercept) const noexcept;

  template <typename T>
  double
  LambdaMax(const T&               x, // samples in rows
            const Eigen::MatrixXd& y,
            const Eigen::ArrayXd&  y_scale) const noexcept;

  double L_scaling;
};

class Gaussian : public Family {
public:
  Gaussian() : Family(1.0) {}

  void
  Preprocess(Eigen::MatrixXd& y,
             Eigen::ArrayXd&  y_center,
             Eigen::ArrayXd&  y_scale) noexcept
  {
    y_center = Mean(y);
    y_scale = StandardDeviation(y, y_center);

    auto n = y.rows();
    for (decltype(n) i = 0; i < n; ++i)
      y(i) = (y(i) - y_center(0))/y_scale(0);
  }

  double
  Loss(const Eigen::ArrayXd&  linear_predictor,
       const Eigen::MatrixXd& y,
       const unsigned         i) const noexcept
  {
    return 0.5*(linear_predictor(0) - y(i))*(linear_predictor(0) - y(i));
  }

  void
  Gradient(const Eigen::ArrayXd&  linear_predictor,
           const Eigen::MatrixXd& y,
           const unsigned         i,
           Eigen::ArrayXd&        gradient) const noexcept
  {
    gradient(0) = linear_predictor(0) - y(i);
  }

  double
  NullDeviance(const Eigen::MatrixXd& y,
               const bool             fit_intercept) const noexcept
  {
    Eigen::ArrayXd linear_predictor = Mean(y.transpose());

    double loss = 0.0;
    auto n = y.cols();
    for (decltype(n) i = 0; i < n; ++i)
      loss += Loss(linear_predictor, y, i);

    return 2.0 * loss;
  }

  void
  FitNullModel(const Eigen::MatrixXd& y,
               const bool             fit_intercept,
               Eigen::ArrayXd&        intercept) {
    intercept = Mean(y.transpose());
  }

  template <typename T>
  double
  LambdaMax(const T&               x,
            const Eigen::MatrixXd& y,
            const Eigen::ArrayXd&  y_scale) const noexcept
  {
    return y_scale(0)*(x.transpose() * y).cwiseAbs().maxCoeff()/x.rows();
  }
};

class Binomial : public Family {
public:
  Binomial() : Family(0.25) {}

  void
  Preprocess(Eigen::MatrixXd& y,
             Eigen::ArrayXd&  y_center,
             Eigen::ArrayXd&  y_scale) const noexcept
  {
    // no preprocessing
  }

  double
  Link(double y) const noexcept
  {
    // TODO(jolars): let the user set this.
    double pmin = 1e-9;
    double pmax = 1.0 - pmin;
    double z = Clamp(y, pmin, pmax);

    return std::log(z / (1.0 - z));
  }

  double
  Loss(const Eigen::ArrayXd&  linear_predictor,
       const Eigen::MatrixXd& y,
       const unsigned         i) const noexcept
  {
    return std::log(1.0 + std::exp(linear_predictor(0)))
           - y(i)*linear_predictor(0);
  }

  void
  Gradient(const Eigen::ArrayXd&  linear_predictor,
           const Eigen::MatrixXd& y,
           const unsigned         i,
           Eigen::ArrayXd&        gradient) const noexcept
  {
    gradient(0) = 1.0 - y(i) - 1.0/(1.0 + std::exp(linear_predictor(0)));
  }

  double
  NullDeviance(const Eigen::MatrixXd& y,
               const bool             fit_intercept) const noexcept
  {
    Eigen::ArrayXd linear_predictor(1);

    if (fit_intercept) {
      auto y_bar = Mean(y.transpose());
      linear_predictor(0) = Link(y_bar(0));
    } else {
      linear_predictor(0) = 0.0;
    }

    double loss = 0.0;
    for (unsigned i = 0; i < y.cols(); ++i)
      loss += Loss(linear_predictor, y, i);

    return 2.0 * loss;
  }

  void
  FitNullModel(const Eigen::MatrixXd& y,
               const bool             fit_intercept,
               Eigen::ArrayXd&        intercept) {

    if (fit_intercept) {
      auto y_bar = Mean(y.transpose());
      intercept(0) = Link(y_bar(0));
    } else {
      intercept(0) = 0.0;
    }
  }

  template <typename T>
  double
  LambdaMax(const T&               x,
            const Eigen::MatrixXd& y,
            const Eigen::ArrayXd&  y_scale) const noexcept
  {
    auto n = y.rows();

    Eigen::VectorXd y_map(n);

    auto y_bar = Mean(y);
    auto y_std = StandardDeviation(y, y_bar);

    for (decltype(n) i = 0; i < n; ++i)
      y_map(i) = (y(i) - y_bar(0))/y_std(0);

    return y_std(0)*(x.transpose() * y_map).cwiseAbs().maxCoeff()/n;
  }
};

class Multinomial : public Family {
public:
  Multinomial(const unsigned n_classes) : Family(0.25), n_classes(n_classes) {}

  void
  Preprocess(Eigen::MatrixXd& y,
             Eigen::ArrayXd&  y_center,
             Eigen::ArrayXd&  y_scale) const noexcept
  {
    // no preprocessing
  }

  double
  Loss(const Eigen::ArrayXd&  linear_predictor,
       const Eigen::MatrixXd& y,
       const unsigned         i) const noexcept
  {
    auto c = static_cast<unsigned>(y(i) + 0.5);
    return LogSumExp(linear_predictor) - linear_predictor[c];
  }

  void
  Gradient(const Eigen::ArrayXd&  linear_predictor,
           const Eigen::MatrixXd& y,
           const unsigned         i,
           Eigen::ArrayXd&        gradient) const noexcept
  {
    auto lse = LogSumExp(linear_predictor);
    unsigned p = linear_predictor.size();
    auto c = static_cast<unsigned>(y(i) + 0.5);

    for (decltype(p) j = 0; j < p; ++j) {
      gradient[j] = std::exp(linear_predictor[j] - lse);

      if (j == c)
        gradient[j] -= 1.0;
    }
  }

  double
  NullDeviance(const Eigen::MatrixXd& y,
               const bool             fit_intercept) const noexcept
  {
    Eigen::ArrayXd linear_predictor(n_classes);

    if (fit_intercept)
      linear_predictor = Proportions(y, n_classes);
    else
      linear_predictor.setConstant(1.0/n_classes);

    linear_predictor =
      linear_predictor.log() - linear_predictor.log().sum()/n_classes;

    auto lse = LogSumExp(linear_predictor);

    auto loss = 0.0;
    for (decltype(y.cols()) i = 0; i < y.cols(); ++i) {
      auto c = static_cast<unsigned>(y(i) + 0.5);
      loss += lse - linear_predictor[c];
    }

    return 2.0 * loss;
  }

  void
  FitNullModel(const Eigen::MatrixXd& y,
               const bool             fit_intercept,
               Eigen::ArrayXd&        intercept) {

    if (fit_intercept)
      intercept = Proportions(y, n_classes);
    else
      intercept = 1.0/n_classes;

    intercept = intercept.log() - intercept.log().sum()/n_classes;
  }

  template <typename T>
  double
  LambdaMax(const T&               x,
            const Eigen::MatrixXd& y,
            const Eigen::ArrayXd&  y_scale) const noexcept
  {
    auto n = y.rows();

    Eigen::MatrixXd y_map = Eigen::MatrixXd::Zero(n, n_classes);

    for (decltype(n) i = 0; i < n; ++i) {
      auto c = static_cast<unsigned>(y(i) + 0.5);
      y_map(i, c) = 1.0;
    }

    auto y_bar = Mean(y_map);
    auto y_std = StandardDeviation(y_map, y_bar);
    Standardize(y_map, y_bar, y_std);

    Eigen::ArrayXXd inner_products = x.transpose() * y_map;

    for (decltype(inner_products.cols()) k = 0; k < inner_products.cols(); ++k)
      inner_products.col(k) *= y_std(k);

    return inner_products.abs().maxCoeff()/n;
  }

private:
  const unsigned n_classes;
};

class MultivariateGaussian : public Family {
public:
  MultivariateGaussian(const bool standardize_response)
                       : Family(1.0),
                         standardize_response(standardize_response) {}

  void
  Preprocess(Eigen::MatrixXd& y,
             Eigen::ArrayXd&  y_center,
             Eigen::ArrayXd&  y_scale) const noexcept
  {
    if (standardize_response) {
      // NOTE(jolars): this is the kind of standardization that glmnet does,
      // i.e. it standardizes y but does not recover unstandardized versions
      // of the coefficients.
      Standardize(y);
    }
  }

  double
  Loss(const Eigen::ArrayXd&  linear_predictor,
       const Eigen::MatrixXd& y,
       const unsigned         i) const noexcept
  {
    return 0.5*(linear_predictor - y.array().col(i)).square().sum();
  }

  void
  Gradient(const Eigen::ArrayXd&  linear_predictor,
           const Eigen::MatrixXd& y,
           const unsigned         i,
           Eigen::ArrayXd&        gradient) const noexcept
  {
    gradient = linear_predictor - y.array().col(i);
  }

  double
  NullDeviance(const Eigen::MatrixXd& y,
               const bool             fit_intercept) const noexcept
  {
    auto linear_predictor = Mean(y.transpose());

    double loss = 0.0;
    for (decltype(y.cols()) i = 0; i < y.cols(); ++i)
      loss += Loss(linear_predictor, y, i);

    return 2.0 * loss;
  }

  void
  FitNullModel(const Eigen::MatrixXd& y,
               const bool             fit_intercept,
               Eigen::ArrayXd&        intercept) {
    intercept = Mean(y.transpose());
  }

  template <typename T>
  double
  LambdaMax(const T&               x,
            const Eigen::MatrixXd& y,
            const Eigen::ArrayXd&  y_scale) const noexcept
  {
    Eigen::MatrixXd y_map(y);

    auto y_bar = Mean(y);
    auto y_std = StandardDeviation(y, y_bar);

    Standardize(y_map, y_bar, y_std);

    Eigen::ArrayXXd inner_products = x.transpose() * y_map;

    for (decltype(inner_products.cols()) k = 0; k < inner_products.cols(); ++k)
      inner_products.col(k) *= y_scale(k)*y_std(k);

    return inner_products.square().rowwise().sum().sqrt().maxCoeff()/y.rows();
  }

private:
  const bool standardize_response;
};

class Poisson : public Family {
public:
  Poisson() : Family(1.0) {}

  void
  Preprocess(Eigen::MatrixXd& y,
             Eigen::ArrayXd&  y_center,
             Eigen::ArrayXd&  y_scale) const noexcept
  {
    // no preprocessing
  }

  double
  Loss(const Eigen::ArrayXd&  linear_predictor,
       const Eigen::MatrixXd& y,
       const unsigned         i) const noexcept
  {
    return std::exp(linear_predictor(0)) - y(i)*linear_predictor(0);
  }

  void
  Gradient(const Eigen::ArrayXd&  linear_predictor,
           const Eigen::MatrixXd& y,
           const unsigned         i,
           Eigen::ArrayXd&        gradient) const noexcept
  {
    gradient(0) = std::exp(linear_predictor(0)) - y(i);
  }

  double
  NullDeviance(const Eigen::MatrixXd& y,
               const bool             fit_intercept) const noexcept
  {
    Eigen::ArrayXd linear_predictor(1);

    if (fit_intercept) {
      auto y_bar = Mean(y.transpose());
      linear_predictor(0) = std::log(y_bar(0));
    } else {
      linear_predictor(0) = 0.0;
    }

    double loss = 0.0;
    for (unsigned i = 0; i < y.cols(); ++i){
      if (y(i) != 0)
        loss +=  y(i)*std::log(y(i)) - y(i)*linear_predictor(0);
    }
    return 2.0 * loss;
  }

  void
  FitNullModel(const Eigen::MatrixXd& y,
               const bool             fit_intercept,
               Eigen::ArrayXd&        intercept) {

    if (fit_intercept) {
      auto y_bar = Mean(y.transpose());
      intercept(0) = std::log(y_bar(0));
    } else {
      intercept(0) = 0.0;
    }
  }

  template <typename T>
  double
    LambdaMax(const T&               x,
              const Eigen::MatrixXd& y,
              const Eigen::ArrayXd&  y_scale) const noexcept
    {
      auto n = y.rows();

      Eigen::VectorXd y_map(n);

      auto y_bar = Mean(y);
      auto y_std = StandardDeviation(y, y_bar);

      for (decltype(n) i = 0; i < n; ++i)
        y_map(i) = (y(i) - y_bar(0))/y_std(0);

      return y_std(0)*(x.transpose() * y_map).cwiseAbs().maxCoeff()/n;
    }

};

} // namespace sgdnet

#endif // SGDNET_FAMILIES_
