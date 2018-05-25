#ifndef SGDNET_FAMILIES_
#define SGDNET_FAMILIES_

#include <RcppArmadillo.h>
#include <memory>

namespace sgdnet {

class Family {
public:
  virtual ~Family() {};

  virtual double Loss(const arma::rowvec& prediction,
                      const arma::rowvec& y) = 0;

  virtual arma::rowvec Gradient(const arma::rowvec& prediction,
                                const arma::rowvec& y) = 0;

  virtual arma::uword NClasses(const arma::mat& y) = 0;

  virtual arma::vec StepSize(const double       max_squared_sum,
                             const arma::vec&   alpha_scaled,
                             const bool         fit_intercept,
                             const arma::uword  n_samples) = 0;
};

class Gaussian : public Family {
public:
  virtual ~Gaussian() {};

  double Loss(const arma::rowvec& prediction,
              const arma::rowvec& y) {
    return 0.5*arma::accu(arma::square(prediction - y));
  };

  arma::rowvec Gradient(const arma::rowvec& prediction,
                        const arma::rowvec& y) {
    return prediction - y;
  };

  arma::uword NClasses(const arma::mat& y) {
    return 1;
  }

  arma::vec StepSize(const double       max_squared_sum,
                     const arma::vec&   alpha_scaled,
                     const bool         fit_intercept,
                     const arma::uword  n_samples) {
    // Lipschitz constant approximation
    arma::vec L = (max_squared_sum + fit_intercept) + alpha_scaled;
    arma::vec mu_n = 2.0*n_samples*alpha_scaled;
    return 1.0 / (2.0*L + arma::min(L, mu_n));
  };
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
