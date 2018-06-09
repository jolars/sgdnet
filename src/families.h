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

namespace sgdnet {

class Family {
public:
  virtual ~Family() {};

  virtual double Loss(const arma::mat& prediction,
                      const arma::mat& y) = 0;

  virtual arma::rowvec Gradient(const arma::rowvec& prediction,
                                const arma::rowvec& y) = 0;

  virtual arma::uword NClasses(const arma::mat& y) = 0;

  virtual arma::vec StepSize(const double       max_squared_sum,
                             const arma::vec&   alpha_scaled,
                             const bool         fit_intercept,
                             const arma::uword  n_samples) = 0;

  virtual void PreprocessResponse(arma::mat&    y,
                                  arma::rowvec& y_center,
                                  arma::rowvec& y_scale,
                                  const bool    fit_intercept) = 0;

  virtual double Deviance(const arma::mat& y,
                          const arma::mat& prediction) = 0;
};

class Gaussian : public Family {
public:
  virtual ~Gaussian() {};

  double Loss(const arma::mat& prediction,
              const arma::mat& y) {
    return 0.5*arma::accu(arma::square(prediction - y));
  };

  arma::rowvec Gradient(const arma::rowvec& prediction,
                        const arma::rowvec& y) {
    return prediction - y;
  };

  arma::uword NClasses(const arma::mat& y) {
    return 1;
  };

  arma::vec StepSize(const double       max_squared_sum,
                     const arma::vec&   alpha_scaled,
                     const bool         fit_intercept,
                     const arma::uword  n_samples) {
    // Lipschitz constant approximation
    arma::vec L = (max_squared_sum + fit_intercept) + alpha_scaled;
    arma::vec mu_n = 2.0*n_samples*alpha_scaled;
    return 1.0 / (2.0*L + arma::min(L, mu_n));
  };

  void PreprocessResponse(arma::mat&    y,
                          arma::rowvec& y_center,
                          arma::rowvec& y_scale,
                          const bool    fit_intercept) {

    if (fit_intercept) {
      y_center = arma::mean(y);
      y_scale  = arma::sqrt(arma::var(y, 1));

      for (arma::uword i = 0; i < y.n_cols; ++i) {
        y.col(i) -= y_center(i);
        if (y_scale(i) != 0)
          y.col(i) /= y_scale(i);
      }
    } else {
      y_center.zeros();
      y_scale.ones();
    }
  };

  // Returns the fraction of deviance explained
  double Deviance(const arma::mat& prediction,
                  const arma::mat& y) {
    return 1.0 -
      arma::accu(arma::square(prediction - y))/arma::accu(arma::square(y));
  }
};

class FamilyFactory {
public:
  static std::unique_ptr<Family> NewFamily(const std::string& family_choice) {
    if (family_choice == "gaussian")
      return std::unique_ptr<Family>(new Gaussian());
    return NULL;
  };
};

} // namespace sgdnet

#endif // SGDNET_FAMILIES_
