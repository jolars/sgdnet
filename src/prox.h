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

#ifndef SGDNET_PROX_
#define SGDNET_PROX_

#include <memory>

namespace sgdnet {

//' Soft thresholding operator for L1-regularization
//'
//' Solves \f$ \argmin_{x} 0.5||x - y||^{2} + \alpha ||x||_{1} \f$.
//'
//' @inheritParams Prox
//'
//' @noRd
//' @keywords internal
struct SoftThreshold {
  template <typename T>
  T
  operator()(const T& x, const T& shrinkage) const noexcept
  {
    return std::max(x - shrinkage, 0.0) - std::max(-x - shrinkage, 0.0);
  }
};

}

#endif // SGDNET_PROX_
