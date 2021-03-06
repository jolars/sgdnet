% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/deviance.sgdnet.R
\name{deviance.sgdnet}
\alias{deviance.sgdnet}
\alias{deviance.cv_sgdnet}
\title{Deviance for sgdnet Object}
\usage{
\method{deviance}{sgdnet}(object, ...)

\method{deviance}{cv_sgdnet}(object, ...)
}
\arguments{
\item{object}{an object of class \code{'sgdnet'}}

\item{...}{ignored}
}
\value{
The deviance of \code{object} at each value along the regularization path.
For \code{family = "gaussian"} in \code{\link[=sgdnet]{sgdnet()}}, this is the residual sum of
squares.
}
\description{
Return the deviance for an object of class \code{"sgdnet"}, typically
from a fit with \code{\link[=sgdnet]{sgdnet()}}. \code{deviance.cv_sgdnet()} is a simple wrapper
that calls \code{\link[=deviance.sgdnet]{deviance.sgdnet()}} on the fit to full data from \code{\link[=cv_sgdnet]{cv_sgdnet()}}.
}
\details{
This functions returns the deviance of the model along the regularization
path. It is computed from the slots \code{dev.ratio} and \code{nulldev} from the
\code{"sgdnet"} object using the formula

\deqn{
  (1 - \mathtt{dev.ratio}) \times \mathtt{nulldev}
}{
  (1 - dev.ratio)*nulldev
}

where \code{nulldev} is the deviance of the intercept-only model.
}
\examples{
fit <- sgdnet(wine$x, wine$y, family = "multinomial")
deviance(fit)
}
\seealso{
\code{\link[stats:deviance]{stats::deviance()}}, \code{\link[=sgdnet]{sgdnet()}}
}
