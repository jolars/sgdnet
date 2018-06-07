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
inline double LogSumExp(const std::vector<double>& x) {
  double x_max = *std::max_element(x.begin(), x.end());
  double exp_sum = 0.0;

  for (auto const& x_val : x)
    exp_sum += std::exp(x_val - x_max);

  return std::log(exp_sum) + x_max;
}

//' Log-spaced sequence
//'
//' @param from starting number
//' @param to finishing number
//' @param n number of elements
//'
//' @return a log-spaced sequence
inline std::vector<double> LogSpace(const double      from,
                                    const double      to,
                                    const std::size_t n) {
  double log_from = std::log(from);
  double step = (std::log(to) - log_from)/(n - 1);

  std::vector<double> out;
  out.reserve(n);

  for (std::size_t i = 0; i < n; ++i)
    out.push_back(std::exp(log_from + i*step));

  return out;
}

template <typename T>
inline double Mean(const T& x, const std::size_t n) {
  return std::accumulate(x.begin(), x.end(), 0.0)/n;
}

template <typename T>
inline double StandardDeviation(const T& x, const std::size_t n) {
  double x_mean = Mean(x, n);

  double squared_deviance = 0.0;

  for (const auto& x_val : x)
    squared_deviance += (x_val - x_mean)*(x_val - x_mean)/n;

  return std::sqrt(squared_deviance);
}

#endif // SGDNET_MATH_
