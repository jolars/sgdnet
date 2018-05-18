#ifndef SGDNET_OBJECTIVES_
#define SGDNET_OBJECTIVES_

#include <RcppArmadillo.h>

namespace sgdnet {
  class Objective {
  public:
    virtual double Loss(const arma::mat& p, const arma::mat& y) = 0;
    virtual arma::mat Gradient(const arma::mat& p, const arma::mat& y) = 0;
  };

  class Gaussian : public Objective {
  public:
    double Loss(const arma::mat& p, const arma::mat& y) {
      return 0.5*arma::accu(arma::square(p - y));
    };
    arma::mat Gradient(const arma::mat& p, const arma::mat& y) {
      return p - y;
    };
  };
}

#endif // SGDNET_OBJECTIVES
