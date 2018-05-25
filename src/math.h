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

#ifndef SGDNET_MATH_
#define SGDNET_MATH_

#include <RcppArmadillo.h>

//' LogSumExp function
//'
//' @param x a vector in the log domain
//'
//' @return `log(sum(exp(x)))` while avoiding over/underflow.
inline double LogSumExp(const arma::vec& x) {
  double x_max = x.max();
  return std::log(arma::accu(arma::exp(x - x_max))) + x_max;
}

#endif // SGDNET_MATH_
