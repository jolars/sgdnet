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
#' fit <- sgdnet(iris[, 1], iris[, 2:4])
#' plot(fit, main = "Lassoing with sgdnet", type = "S")
plot.sgdnet <- function(x, xvar = c("norm", "lambda", "dev"), ...) {
  lambda <- x$lambda
  beta <- t(as.matrix(x$beta))

  plot_data <- utils::stack(as.data.frame(beta))

  plot_args <- list(
    x = quote(values ~ xval),
    groups = quote(ind),
    data = quote(plot_data),
    type = "l",
    ylab = expression(beta),
    auto.key = list(space = "right", lines = TRUE, points = FALSE)
  )

  switch(match.arg(xvar),
         norm = {
           plot_args$xlab <- "L1 Norm"
           plot_data$xval <- rowSums(abs(beta))
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
           plot_args$xlab <- "Deviance"
           plot_data$xval <- x$dev.ratio
         })

  # Let the user modify the plot parameters
  do.call(lattice::xyplot, utils::modifyList(plot_args, list(...)))
}
