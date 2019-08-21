# Load glmnet and set options for it
library(glmnet)
glmnet.control(fdev = 0)

# generate random data simulated from a generalized linear model
random_data <- function(n = 100,
                        p = 2,
                        family = c("gaussian",
                                   "binomial",
                                   "multinomial",
                                   "mgaussian",
                                   "poisson"),
                        intercept = TRUE,
                        density = 0.5) {
  family <- match.arg(family)

  x <- matrix(rnorm(n*p, 0, 0.01), n, p)

  x[sample(n*p, (1 - density)*n*p)] <- 0

  k <- if (family %in% c("multinomial", "mgaussian")) 3 else 1

  beta <- matrix(sample(seq(-1, 1, length.out = k*p*10), p*k, TRUE), k, p)
  if (intercept)
    beta <- cbind(sample(seq(-1, 1, length.out = k*p*10), k, TRUE), beta)

  center <- sample(seq(-0.1, 0.1, length.out = p*10), p)
  scale <- sample(seq(1.05, 0.95, length.out = p*10), p)

  x <- t((t(x)+center)*scale)

  z <- tcrossprod(if (intercept) cbind(1, x) else x, beta)

  switch(
    family,
    gaussian = {
      y <- rnorm(n, z, 0.01)
    },
    binomial = {
      pr <- 1/(1 + exp(-z))
      y <- rbinom(n, 1, pr)
    },
    multinomial = {
      pr <- t(apply(z, 1, function(x) exp(x)/sum(exp(x))))
      y <- apply(pr, 1, function(x) which(as.logical(rmultinom(1, 1, x))))
    },
    mgaussian = {
      y <- matrix(rnorm(n*k, z, 0.01), n, k)
    },
    poisson = {
      mu <- exp(z)
      y <- rpois(n, mu)
    }
  )

  x <- Matrix::Matrix(x, sparse = TRUE)

  list(x = x, y = y)
}

compare_predictions <- function(f1,
                                f2,
                                x,
                                type = c("link",
                                         "response",
                                         "coefficients",
                                         "class"),
                                tol = 1e-3) {
  types <- match.arg(type, several.ok = TRUE)

  alpha <- f1$alpha

  is_glmnet <- inherits(f1, "glmnet") || inherits(f2, "glmnet")

  for (type in types) {
    if (!inherits(f1, c("sgdnet_multinomial", "sgdnet_binomial"))
        && type == "class")
      next

    c1 <- predict(f1, x, type = type)
    c2 <- predict(f2, x, type = type)

    if (is.list(c1)) {
      c1 <- do.call(rbind, c1)
      c2 <- do.call(rbind, c2)
    }

    c1 <- as.matrix(c1)
    c2 <- as.matrix(c2)

    if (is_glmnet && alpha == 0) {
      # in this case we don't check the first coefficients since glmnet
      # uses a different penalty here
      c1 <- c1[, -1, drop = FALSE]
      c2 <- c2[, -1, drop = FALSE]
    }

    if (type == "class") {
      frac_agree <- sum(c1 == c2)/length(c1)
      testthat::expect_gte(frac_agree, 0.95)
    } else {
      testthat::expect_equivalent(c1, c2, tol = tol)
    }
  }
}

# Store the current families
families <- function() {
  c("gaussian", "binomial", "multinomial", "mgaussian", "poisson")
}

# Capture plots without plotting
dont_plot <- function(x, ...) {
  tmp <- tempfile()
  grDevices::png(tmp)
  p <- graphics::plot(x, ...)
  grDevices::dev.off()
  unlink(tmp)
  invisible(p)
}

# Suppress printning
dont_print <- function(x, ...) {
  utils::capture.output(y <- print(x, ...))
  invisible(y)
}
