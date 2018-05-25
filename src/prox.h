#ifndef SGDNET_PROX_
#define SGDNET_PROX_

#include <memory>

namespace sgdnet {

//' Base class for proximal operators
//'
//' @param x value
//' @param shrinkage shrinkage
//'
//' @noRd
//' @keywords internal
class Prox {
public:
  virtual double Evaluate(const double x, const double shrinkage) = 0;

  virtual ~Prox() {};
};

//' Soft thresholding operator for L1-regularization
//'
//' Solves \f$ \argmin_{x} 0.5||x - y||^{2} + \alpha ||x||_{1} \f$.
//'
//' @inheritParams Prox
//'
//' @noRd
//' @keywords internal
class SoftThreshold : public Prox {
public:
  double Evaluate(const double x, const double shrinkage) {
    return std::max(x - shrinkage, 0.0) - std::max(-x - shrinkage, 0.0);
  }

  virtual ~SoftThreshold() {};
};

class ProxFactory {
public:
  static std::unique_ptr<Prox> NewProx(const std::string& prox_choice) {
    if (prox_choice == "soft_threshold")
      return std::unique_ptr<Prox>(new SoftThreshold());
    return NULL;
  };
};

} // namespace sgdnet

#endif // SGDNET_PROX_
