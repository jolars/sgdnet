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
inline
double
LogSumExp(const T& x) {
  auto x_max = *std::max_element(x.begin(), x.end());
  auto exp_sum = 0.0;

  for (auto x_i : x)
    exp_sum += std::exp(x_i - x_max);

  return std::log(exp_sum) + x_max;
}

//' Log-spaced sequence
//'
//' @param from starting number
//' @param to finishing number
//' @param n number of elements
//'
//' @return a log-spaced sequence
inline
std::vector<double>
LogSpace(const double from, const double to, const unsigned n) {

  auto log_from = std::log(from);
  auto step = (std::log(to) - log_from)/(n - 1);

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
inline
std::vector<double>
Mean(const T& x) {
  auto n = x.rows();
  auto m = x.cols();

  std::vector<double> x_bar;
  x_bar.reserve(m);

  for (decltype(m) j = 0; j < m; ++j)
    x_bar.emplace_back(x.col(j).sum()/n);

  return x_bar;
}

//' Standard deviation
//'
//' @param x vector-like container
//' @param n number of samples
//'
//' @return The arithmetic mean.
//'
//' @noRd
inline
std::vector<double>
StandardDeviation(const Eigen::SparseMatrix<double>& x,
                  const std::vector<double>&         x_bar) {
  auto n = x.rows();
  auto m = x.cols();

  std::vector<double> x_std(m);

  for (decltype(m) j = 0; j < m; ++j) {

    double var = 0.0;
    for (Eigen::SparseMatrix<double>::InnerIterator x_itr(x, j); x_itr; ++x_itr)
      var += std::pow(x_itr.value() - x_bar[j], 2)/n;

    auto n_zeros = n - x.col(j).nonZeros();
    var += n_zeros*x_bar[j]*x_bar[j]/n;

    x_std[j] = (var == 0.0) ? 1.0 : std::sqrt(var);
  }

  return x_std;
}

inline
std::vector<double>
StandardDeviation(const Eigen::MatrixXd&     x,
                  const std::vector<double>& x_bar) {
  auto n = x.rows();
  auto m = x.cols();

  std::vector<double> x_std(m);

  for (decltype(m) j = 0; j < m; ++j) {
    double var = (x.col(j).array() - x_bar[j]).square().sum()/n;
    x_std[j] = (var == 0.0) ? 1.0 : std::sqrt(var);
  }

  return x_std;
}

template <typename T>
inline
std::vector<double> StandardDeviation(const T& x) {
  return StandardDeviation(x, Mean(x));
}

template <typename T>
inline
void
Standardize(T&                         x,
            const std::vector<double>& x_bar,
            const std::vector<double>& x_std) {
  auto m = x.cols();

  for (decltype(m) j = 0; j < m; ++j)
    x.col(j) = (x.col(j).array() - x_bar[j])/x_std[j];
}

template <typename T>
inline
void
Standardize(T& x) {
  auto x_bar = Mean(x);
  auto x_std = StandardDeviation(x, x_bar);
  Standardize(x, x_bar, x_std);
}

//' Clamp a value to [min, max]
//' @param x value to clamp
//' @param min min
//' @param max max
//' @noRd
template <typename T>
inline
T
Clamp(const T& x, const T& min, const T& max) {
  return x > max ? max : (x < min ? min : x);
}

//' Table proportions of unique values in vector
//'
//' The input vector `x` will be coerced into unsigned integers by
//' static_cast<unsigned>()
//'
//' @param x vector with integerish components
//' @param n_classes number of classes (unique values)
//'
//' @return A
//' @noRd
template <typename T>
inline
std::vector<double>
Proportions(const T& y, const unsigned n_classes) {
  std::vector<double> proportions(n_classes);
  auto n = y.cols();

  for (decltype(n) i = 0; i < n; ++i) {
    auto c = static_cast<unsigned>(y(0, i) + 0.5);
    proportions[c] += 1.0/n;
  }

  return proportions;
}

#endif // SGDNET_MATH_
