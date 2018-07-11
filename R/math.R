# sgdnet: Penalized Generalized Linear Models with Stochastic Gradient Descent
# Copyright (C) 2018  Johan Larsson
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

softmax <- function(x) {
  d <- dim(x)
  nas <- apply(is.na(x), 1, any)
  if (any(nas)) {
    pclass <- rep(NA, d[1])
    if (sum(nas) < d[1]) {
      pclass2 <- softmax(x[!nas, ])
      pclass[!nas] <- pclass2
      if (is.factor(pclass2))
        pclass <- factor(pclass, levels = seq(d[2]), labels = levels(pclass2))
    }
  } else {
    maxdist <- x[, 1]
    pclass <- rep(1, d[1])
    for (i in seq(2, d[2])) {
      l <- x[, i] > maxdist
      pclass[l] <- i
      maxdist[l] <- x[l, i]
    }
    dd <- dimnames(x)[[2]]
    pclass <- if (is.null(dd) || !length(dd))
      pclass
    else
      factor(pclass, levels = seq(d[2]), labels = dd)
  }
  pclass
}

