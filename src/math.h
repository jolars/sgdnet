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

//' LogSumExp function
//'
//' @param x a vector in the log domain
//'
//' @return `log(sum(exp(x)))` while avoiding over/underflow.
template <typename T>
inline double LogSumExp(const T& x) {
  auto x_max = std::max_element(x.begin(), x.end());
  double exp_sum = 0.0;

  for (auto x_i : x)
    exp_sum += std::exp(x_i - (*x_max));

  return std::log(exp_sum) + (*x_max);
}

//' Log-spaced sequence
//'
//' @param from starting number
//' @param to finishing number
//' @param n number of elements
//'
//' @return a log-spaced sequence
inline std::vector<double> LogSpace(const double from,
                                    const double to,
                                    const int    n) {
  double log_from = std::log(from);
  double step = (std::log(to) - log_from)/(n - 1);

  std::vector<double> out;
  out.reserve(n);

  for (unsigned i = 0; i < n; ++i)
    out.push_back(std::exp(log_from + i*step));

  return out;
}

//' Artithmetic mean of vector
//'
//' @param x vector-like object
//' @param n number of samples
//'
//' @return The arithmethic mean.
//'
//' @noRd
template <typename T>
inline double Mean(const T& x) {
  return std::accumulate(x.begin(), x.end(), 0.0)/x.size();
}

//' Standard deviation
//'
//' @param x vector-like container
//' @param n number of samples
//'
//' @return The arithmetic mean.
//'
//' @noRd
template <typename T>
inline double StandardDeviation(const T& x, const double x_mean) {
  double var = 0.0;
  for (auto x_i : x)
    var += std::pow(x_i - x_mean, 2)/x.size();

  // Never return 0 standard deviation
  return var == 0.0 ? 1.0 : std::sqrt(var);
}

template <typename T>
inline double StandardDeviation(const T& x) {
  return StandardDeviation(x, Mean(x));
}

template <typename T>
inline void Standardize(const T& x) {
  auto x_mu = Mean(x);
  auto x_sd = StandardDeviation(x, x_mu);

  for (auto& x_i : x) {
    x_i -= x_mu;
    x_i /= x_sd;
  }
}


#endif // SGDNET_MATH_
