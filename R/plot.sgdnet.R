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

#' Plot Coefficients from an 'sgdnet' Object
#'
#' Plot coefficients from an object of class `'sgdnet'` against the L1-norm,
#' lambda penalty, or deviance ratio.
#'
#' This function calls [lattice::xyplot()] under the hood after having
#' arranged the plotting data slightly.
#'
#' @param x an object of class `'sgdnet'`, commonly the result from calling
#'   [sgdnet()].
#' @param xvar value to be plotted on the x axis. `"norm"` plots the
#'   L1 norm, `"lambda"` the logarithmized lambda (penalty) values, and `"dev"`
#'   the percent of deviance explained.
#' @param ... parameters passed down to [lattice::xyplot()].
#'
#' @return A graphical description of class `'trellis'`, which will be
#'   plotted on the current graphical device in interactive sessions.
#' @export
#' @seealso [lattice::xyplot()], [sgdnet()]
#'
#' @examples
#' # Gaussian logistic regression
#' gfit <- sgdnet(abalone$x, abalone$y, alpha = 0)
#' plot(gfit, auto.key = list(columns = 2, space = "top"))
#'
#' # Binomial logistic regression
#' bfit <- sgdnet(with(infert, cbind(age, parity)),
#'                infert$case,
#'                family = "binomial")
#' plot(bfit, xvar = "lambda", grid = TRUE)
#'
#' # Multinomial logistic regression
#' mfit <- sgdnet(iris[, 1:4], iris[, 5], family = "multinomial")
#' plot(mfit, xvar = "dev", main = "Lassoing with sgdnet")
#'
plot.sgdnet <- function(x, xvar = c("norm", "lambda", "dev"), ...) {
  lambda <- x$lambda
  beta <- x$beta
  n_lambda <- length(lambda)

  reorganize <- function(beta) {
    tmp <- t(as.matrix(beta))
    utils::stack(as.data.frame(tmp))
  }

  if (is.list(beta)) {
    n_classes <- length(beta)
    plot_data <- data.frame(matrix(NA, ncol = 3, nrow = 0))
    for (i in seq_len(n_classes)) {
      plot_data <- rbind(plot_data,
                         cbind(reorganize(beta[[i]]), names(beta)[i]))
    }
    colnames(plot_data)[3] <- "response"
  } else {
    plot_data <- reorganize(beta)
  }

  n_vars = length(unique(plot_data$ind))

  plot_args <- list(
    x = if (is.list(beta))
      quote(values ~ xval | response)
    else
      quote(values ~ xval),
    type = if (n_lambda == 1) "p" else "l",
    groups = quote(ind),
    data = quote(plot_data),
    ylab = expression(hat(beta)),
    auto.key = if (n_vars <= 10)
      list(space = "right", lines = TRUE, points = FALSE)
    else FALSE,
    abline = within(lattice::trellis.par.get("reference.line"), {h = 0})
  )

  switch(match.arg(xvar),
         norm = {
           plot_args$xlab <-
             expression(group("|", group("|", hat(beta), "|"), "|")[1])
           plot_data$xval <- if (is.list(beta))
             rowSums(vapply(beta,
                            function(x) colSums(abs(as.matrix(x))),
                            double(ncol(beta[[1]]))))
           else
             colSums(abs(as.matrix(beta)))
         },
         lambda = {
           plot_args$xlab <- expression(lambda)
           plot_args$scales <- list(x = list(log = "e"))
           plot_data$xval <- x$lambda

           # Prettier x scale
           plot_args$xscale.components <- function(lim, ...) {
             x <- lattice::xscale.components.default(lim, ...)
             x$bottom$labels$labels <- parse(text = x$bottom$labels$labels)
             x
           }
         },
         dev = {
           plot_args$xlab <- "Fraction of deviance explained"
           plot_data$xval <- x$dev.ratio
         })

  # Let the user modify the plot parameters
  do.call(lattice::xyplot, utils::modifyList(plot_args, list(...)))
}
