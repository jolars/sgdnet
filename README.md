
<!-- README.md is generated from README.Rmd. Please edit that file -->

# sgdnet

[![Travis build
status](https://travis-ci.org/jolars/sgdnet.svg?branch=master)](https://travis-ci.org/jolars/sgdnet)
[![AppVeyor build
status](https://ci.appveyor.com/api/projects/status/github/jolars/sgdnet?branch=master&svg=true)](https://ci.appveyor.com/project/jolars/sgdnet)
[![Coverage
status](https://codecov.io/gh/jolars/sgdnet/branch/master/graph/badge.svg)](https://codecov.io/github/jolars/sgdnet?branch=master)

**sgdnet** is an R-package that fits elastic net-regularized generalized
linear models to big data using the incremental gradient average
algorithm SAGA (Defazio et al. 2014).

## Installation

**sgdnet** is not currently available on
[CRAN](https://cran.r-project.org/) but can be installed using the
[devtools](https://CRAN.R-project.org/package=devtools) package as
follows:

``` r
# install.packages("devtools")
devtools::install_github("jolars/sgdnet")
```

## Usage

It is simple to fit a model using **sgdnet**. The interface deliberately
mimics that of [glmnet](https://CRAN.R-project.org/package=glmnet) to
facilitate transitionining.

First we load the package, and then we fit a multinomial model to the
[iris](https://en.wikipedia.org/wiki/Iris_flower_data_set) data set. We
se the [elastic net
penalty](https://en.wikipedia.org/wiki/Elastic_net_regularization) to
0.8 using the `alpha` argument to achieve a compromise between the
[ridge](https://en.wikipedia.org/wiki/Tikhonov_regularization) and
[lasso](https://en.wikipedia.org/wiki/Lasso_\(statistics\)) penalties.

**sgdnet** automatically fits the model across an automatically computed
regularization path. Altneratively, the user might supply their own path
using the `lambda` argument.

``` r
library(sgdnet)
fit <- sgdnet(iris[, 1:4], iris[, 5], family = "multinomial", alpha = 0.8)
plot(fit)
```

<img src="man/figures/README-unnamed-chunk-2-1.png" title="The coefficients from a multinomial model along the regularization path fit to the iris data set." alt="The coefficients from a multinomial model along the regularization path fit to the iris data set." width="100%" />

## License

**sgdnet** is open source software, licensed under [GPL-3](LICENSE.md).

## Versioning

**eulerr** uses [semantic versioning](https://semver.org/).

## Acknowledgements

The initial work on **sgdnet** was supported by Google through the
[Google Summer of Code](https://summerofcode.withgoogle.com) program
with Michael Weylandt and Toby Dylan Hocking as mentors.
