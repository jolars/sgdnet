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

#' Plot Results from Cross-Validation
#'
#' @param x an object of class `'cv_sgdnet'`
#' @param sign.lambda the sign of \eqn{\lambda}{lambda}
#' @param ci_alpha alpha (opacity) for fill in confidence limits
#' @param ci_col color for border of confidence limits
#' @param ... other arguments that are passed on to [lattice::xyplot()]
#' @param plot_min whether to mark the location of the lambda corresponding
#'   to the best prediction score
#' @param plot_1se whether to mark the location of the largest lambda
#'   within one standard deviation from the location corresponding to `plot_min`
#' @param ci_border color (or flag to turn off and on) the border of the
#'   confidence limits
#'
#' @seealso [cv_sgdnet], [lattice::xyplot()], [lattice::panel.xyplot()]
#'
#' @return An object of class `'trellis'` is returned and, if used
#'   interactively, will most likely have its print function
#'   [lattice::print.trellis()]) invoked, which draws the plot on the
#'   current display device.
#'
#' @export
#'
#' @examples
#' cv <- cv_sgdnet(heart$x,
#'                 heart$y,
#'                 family = "binomial",
#'                 alpha = c(0, 0.8, 0.9))
#' plot(cv, ci_alpha = 0.3, plot_1se = FALSE)
plot.cv_sgdnet <-
  function(x,
           sign.lambda = 1,
           plot_min = TRUE,
           plot_1se = TRUE,
           ci_alpha = 0.2,
           ci_border = FALSE,
           ci_col = lattice::trellis.par.get("superpose.line")$col,
           ...) {
  fit <- x
  data <- fit$cv_summary
  data$alpha <- as.factor(data$alpha)
  data$lambda <- data$lambda*sign.lambda
  n_alpha <- length(unique(data$alpha))

  if (n_alpha > 1) {
    form <- stats::formula(mean ~ lambda | alpha)
  } else {
    form <- stats::formula(mean ~ lambda)
  }

  optimal_alpha_ind <- match(x$alpha_min, x$alpha)

  args <- list(
    x = form,
    data = data,
    type = "l",
    scales = list(x = list(log = "e", relation = "free")),
    xlab = expression(log[e](lambda)),
    ylab = fit$name,
    grid = FALSE,
    lower = data$ci_lo,
    upper = data$ci_up,
    plot_min = plot_min,
    plot_1se = plot_1se,
    prepanel = prepanel.ci,
    xscale.components = function(lim, ...) {
      x <- lattice::xscale.components.default(lim, ...)
      x$bottom$labels$labels <- parse(text = x$bottom$labels$labels)
      x
    },
    strip = lattice::strip.custom(
      var.name = expression(alpha),
      sep = expression(" = "),
      strip.names = TRUE
    ),
    panel = function(x,
                     y,
                     subscripts,
                     lower,
                     upper,
                     grid,
                     plot_min,
                     plot_1se,
                     ...) {
      if (isTRUE(grid))
        lattice::panel.grid(h = -1, v = -1)

      lattice::panel.polygon(
        c(x, rev(x)),
        c(upper[subscripts],
          rev(lower[subscripts])),
        col = ci_col,
        alpha = ci_alpha,
        border = ci_border
      )

      if (lattice::packet.number() == optimal_alpha_ind) {
        if (plot_min)
          lattice::panel.refline(v = sign.lambda * log(fit$lambda_min),
                                 col = 1,
                                 lty = 2)
        if (plot_1se)
          lattice::panel.refline(v = sign.lambda * log(fit$lambda_1se),
                                 col = 1,
                                 lty = 2)
      }

      lattice::panel.xyplot(x, y, ...)
    }
  )

  args <- utils::modifyList(args, list(...))

  do.call(lattice::xyplot, args)
}

#' Prepanel function for Confidence Intervals
#'
#' @param x x-axis values
#' @param y y-axis values
#' @param lower lower confidence limits
#' @param upper upper confidence limits
#' @param subscripts indices for current panel
#' @param groups groups
#' @param ... ignored
#'
#' @return Sets up limits for a lattice plot.
#' @keywords internal
#' @noRd
prepanel.ci <- function(x,
                        y,
                        lower,
                        upper,
                        subscripts,
                        groups = NULL,
                        ...) {
  if (any(!is.na(x)) && any(!is.na(y))) {
    ord <- order(as.numeric(x))
    if (!is.null(groups)) {
      gg <- groups[subscripts]
      dx <- unlist(lapply(split(as.numeric(x)[ord], gg[ord]), diff))
      dy <- unlist(lapply(split(as.numeric(y)[ord], gg[ord]), diff))
    } else {
      dx <- diff(as.numeric(x[ord]))
      dy <- diff(as.numeric(y[ord]))
    }
    list(xlim = scale.limits(x),
         ylim = scale.limits(c(lower, upper)),
         dx = dx,
         dy = dy,
         xat = if (is.factor(x)) sort(unique(as.numeric(x))) else NULL,
         yat = if (is.factor(y)) sort(unique(as.numeric(y))) else NULL)
  } else {
    list(xlim = rep(NA, 2),
         ylim = rep(NA, 2),
         dx = NA,
         dy = NA)
  }
}


#' Scale Limits
#'
#' This function has been imported from the r package lattice
#' by Deepayan Sarkar. It has ben reformatted slightly.
#'
#' @param x a vector of values, perhaps
#'
#' @return The range (if x is numeric) or levels (if x is a factor).
#'
#' @author Deepayan Sarkar
#' @keywords internal
#' @noRd
scale.limits <- function (x) {
  if (is.factor(x))
    levels(x)
  else if (is.numeric(x))
    range(x, finite = TRUE)
  else
    range(x, na.rm = TRUE)
}
