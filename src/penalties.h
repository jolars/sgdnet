#ifndef SGDNET_PENALTIES_
#define SGDNET_PENALTIES_

#include <RcppEigen.h>
#include "prox.h"

namespace sgdnet {

class Penalty {
public:
  Penalty() : gamma(0.0), alpha(0.0), beta(0.0) {}

  void setParameters(const double gamma_in,
                     const double alpha_in,
                     const double beta_in) noexcept {
    gamma = gamma_in;
    alpha = alpha_in;
    beta = beta_in;
  }

protected:
  double gamma; // step size
  double alpha; // l1 penalty strength
  double beta;  // l2 penalty strength
};

class Ridge : public Penalty  {
public:
  void operator()(Eigen::MatrixXd&       w,
                  const unsigned         j,
                  const double           w_scale,
                  const double           scaling,
                  const Eigen::MatrixXd& g_sum) const noexcept {
    auto p = w.rows();
    for (decltype(p) k = 0; k < p; ++k)
      w(k, j) -= gamma/w_scale*scaling*g_sum(k, j);
  }
};

class ElasticNet : public Penalty  {
public:
  void operator()(Eigen::MatrixXd&       w,
                  const unsigned         j,
                  const double           w_scale,
                  const double           scaling,
                  const Eigen::MatrixXd& g_sum) const noexcept {
    auto p = w.rows();
    for (decltype(p) k = 0; k < p; ++k) {
      w(k, j) -= gamma/w_scale*scaling*g_sum(k, j);
      w(k, j) = prox(w(k, j), beta*gamma*scaling/w_scale);
    }
  }

private:
  sgdnet::SoftThreshold prox{};
};

class GroupLasso : public Penalty  {
public:
  GroupLasso(const unsigned n_classes,
             const unsigned n_features)
             : n_classes(n_classes), n_features(n_features) {}

  void operator()(Eigen::MatrixXd&       w,
                  const unsigned         j,
                  const double           w_scale,
                  const double           scaling,
                  const Eigen::MatrixXd& g_sum) const noexcept {

    for (unsigned k = 0; k < n_classes; ++k) {
      w(k, j) -= gamma/w_scale*scaling*g_sum(k, j);
    }

    auto norm = w.col(j).norm();

    if (norm > beta*gamma*scaling)
      w.col(j).array() -= beta*gamma*scaling/w_scale*w.col(j).array()/norm;
    else
      w.col(j).setConstant(0.0);
  }

private:
  unsigned n_features;
  unsigned n_classes;
  sgdnet::SoftThreshold prox{};
};

} // namespace sgdnet

#endif /* SGDNET_PENALTIES_ */
