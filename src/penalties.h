#ifndef SGDNET_PENALTIES_
#define SGDNET_PENALTIES_

#include <RcppEigen.h>
#include "prox.h"

namespace sgdnet {

class Penalty {
public:
  void
  setParameters(const double gamma_in,
                const double alpha_in,
                const double beta_in,
                const double noncov_in) noexcept
  {
    gamma = gamma_in;
    alpha = alpha_in;
    beta = beta_in;
    noncov = noncov_in;
  }

protected:
  double gamma = 0.0; // step size
  double alpha = 0.0; // l1 penalty strength
  double beta  = 0.0; // l2 penalty strength
  double noncov = 0.0; // non-convexity parameter
};

class Ridge : public Penalty  {
public:
  void
  operator()(Eigen::ArrayXXd&       w,
             const unsigned         j,
             const double           w_scale,
             const double           scaling,
             const Eigen::ArrayXXd& g_sum)
  const noexcept
  {
    w.col(j) -= gamma/w_scale*scaling*g_sum.col(j);
  }
};

class ElasticNet : public Penalty  {
public:
  void
  operator()(Eigen::ArrayXXd&       w,
             const unsigned         j,
             const double           w_scale,
             const double           scaling,
             const Eigen::ArrayXXd& g_sum) const noexcept
  {
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
  void
  operator()(Eigen::ArrayXXd&       w,
             const unsigned         j,
             const double           w_scale,
             const double           scaling,
             const Eigen::ArrayXXd& g_sum) const noexcept
  {
    w.col(j) -= gamma/w_scale*scaling*g_sum.col(j);

    auto factor = beta*gamma*scaling/w.matrix().col(j).norm();

    if (factor < 1.0)
      w.col(j) *= 1.0 - factor/w_scale;
    else
      w.col(j) = 0.0;
  }
};

class MCP : public Penalty {
public:
  void
  operator()(Eigen::ArrayXXd&       w,
             const unsigned         j,
             const double           w_scale,
             const double           scaling,
             const Eigen::ArrayXXd& g_sum) const noexcept
  {
    w.col(j) -= gamma*scaling*g_sum.col(j);

    double l1 = w.matrix().col(j).norm();
 
    if (l1 <= noncov*beta*(1.0 + gamma*alpha)) {
      auto factor = beta*gamma*scaling/w.matrix().col(j).norm();

      if (factor < 1.0)
        w.col(j) *= (1.0 - factor)/(1.0 + alpha*gamma - gamma/noncov);
      else
        w.col(j) = 0.0;

    } else { w.col(j) *= 1.0/(1.0 + alpha*gamma); }

  }

};

class SCAD : public Penalty {
public:
  void
  operator()(Eigen::ArrayXXd&       w,
             const unsigned         j,
             const double           w_scale,
             const double           scaling,
             const Eigen::ArrayXXd& g_sum) const noexcept
  {

    w.col(j) -= gamma*scaling*g_sum.col(j);

    double l1 = w.matrix().col(j).norm();
    auto factor = beta*gamma*scaling/w.matrix().col(j).norm();

    if (l1 <= (beta+gamma*beta*(1.0 + alpha))) {

      if (factor < 1.0)
        w.col(j) *= (1.0 - factor)/(1.0 + alpha*gamma);
      else
        w.col(j) = 0.0;

    }

    else if (l1 <= noncov*beta*(1.0 + alpha*gamma)) {
      factor *= noncov/(noncov - 1.0);

      if (factor < 1.0) {
        w.col(j) *= (1.0 - factor)/(1.0 + alpha*gamma-gamma/(noncov - 1.0));
      }

      else
        w.col(j) = 0.0;
    } else { w.col(j) *= 1.0/(1.0 + alpha*gamma); }

  }
};

} // namespace sgdnet

#endif /* SGDNET_PENALTIES_ */
