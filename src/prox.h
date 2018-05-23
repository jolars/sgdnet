#ifndef SGDNET_PROX_
#define SGDNET_PROX_

namespace sgdnet {

//! Base class for proximal operators
class Prox {
public:
  virtual double Evaluate(const double x, const double shrinkage) = 0;
};

//! Soft thresholding operator for L1-regularization
class SoftThreshold : public Prox {
public:
  double Evaluate(const double x, const double shrinkage) {
    return std::max(x - shrinkage, 0.0) - std::max(-x - shrinkage, 0.0);
  }
};

} // namespace sgdnet

#endif // SGDNET_PROX_
