
# Preprepared benchmark data for vignette ---------------------------------

# Normalize, cut, and aggregate benchmark data
aggregate_benchmarks <- function(x) {
  library(tidyverse)
  x <- x %>%
    group_by(package, penalty, dataset) %>%
    mutate(min_pkg_time = min(time),
           max_pkg_time = max(time)) %>%
    ungroup() %>%
    group_by(dataset, penalty) %>%
    mutate(min_set_time = max(min_pkg_time),
           max_set_time = min(max_pkg_time)) %>%
    ungroup() %>%
    filter(time >= min_set_time, time <= max_set_time) %>%
    mutate(time_rel = (time - min_set_time)/(max_set_time - min_set_time)) %>%
    mutate(time_cut = cut(time_rel, seq(0, 1.0, by = 0.05))) %>%
    group_by(dataset, penalty) %>%
    mutate(min_loss = min(loss),
           max_loss = max(loss)) %>%
    ungroup() %>%
    mutate(normalized_loss = (loss - min_loss)/(max_loss - min_loss)) %>%
    group_by(dataset, penalty, package, time_cut) %>%
    summarise(normalized_median_loss = median(normalized_loss)) %>%
    mutate(time = as.numeric(time_cut)/20) %>%
    select(dataset = dataset,
           penalty = penalty,
           package = package,
           time = time,
           loss = normalized_median_loss)
  x
}

benchmark <- function(datasets, family, n = 1000) {
  library(sgdnet)
  library(glmnet)
  library(SparseM)
  library(Matrix)

  glmnet.control(mnlam = 1)

  # setup tolerance sequence to iterate over
  sgdnet_tol <- signif(exp(seq(log(0.9), log(1e-3), length.out = n)), 2)
  glmnet_tol <- signif(exp(seq(log(0.9), log(1e-9), length.out = n)), 2)

  data <-  data.frame(dataset = character(),
                      package = character(),
                      penalty = character(),
                      time = double(),
                      loss = double())

  for (i in seq_along(datasets)) {
    cat(names(datasets)[i], "\n")
    X <- datasets[[i]]$x
    y <- datasets[[i]]$y
    n_obs <- nrow(X)

    lambda <- 1/n_obs

    for (j in seq_len(n)) {
      set.seed(j*i)
      cat("\r", j, "/", n)
      flush.console()
      if (j == n)
        cat("\n")

      for (penalty in c("ridge", "lasso")) {
        alpha <- ifelse(penalty == "ridge", 0, 1)
        glmnet_time <- system.time({
          tryCatch({
            glmnet_fit <- glmnet::glmnet(X,
                                         y,
                                         family = family,
                                         lambda = lambda,
                                         alpha = alpha,
                                         intercept = TRUE,
                                         standardize = FALSE,
                                         thresh = glmnet_tol[j],
                                         maxit = 1e8)
          },
          error = NA)
        })

        sgdnet_time <- system.time({
          sgdnet_fit <- sgdnet::sgdnet(X,
                                       y,
                                       family = family,
                                       lambda = lambda,
                                       alpha = alpha,
                                       intercept = TRUE,
                                       standardize = FALSE,
                                       thresh = sgdnet_tol[j],
                                       maxit = 1e8)
        })

        glmnet_loss <- deviance(glmnet_fit)
        sgdnet_loss <- deviance(sgdnet_fit)

        data <- rbind(data,
                      data.frame(
                        dataset = names(datasets)[i],
                        package = c("glmnet", "sgdnet"),
                        time = c(glmnet_time[3], sgdnet_time[3]),
                        loss = c(glmnet_loss, sgdnet_loss),
                        penalty = penalty
                      ))
      }
    }
  }
  data
}

# Gaussian ----------------------------------------------------------------

library(libsvmdata) # https://github.com/jolars/libsvmdata/

abalone <- getData("abalone", scaled = TRUE)
cadata <- getData("cadata")
cadata$x <- scale(cadata$x)
cpusmall <- getData("cpusmall", scaled = TRUE)

gaussian_datasets <- list(abalone = abalone,
                          cadata = cadata,
                          cpusmall = cpusmall)

benchmark_gaussian <- benchmark(gaussian_datasets, "gaussian")
benchmark_aggregated_gaussian <- aggregate_benchmarks(benchmark_gaussian)

# Binomial ----------------------------------------------------------------

mushrooms <- getData("mushrooms")
a9a <- getData("a9a", "training")
ijcnn1 <- getData("ijcnn1", "training", scaled = TRUE)
ijcnn1$x <- scale(ijcnn1$x)

binomial_datasets <- list(mushrooms = mushrooms,
                          adult = a9a,
                          ijcnn1 = ijcnn1)

benchmark_binomial <- benchmark(binomial_datasets, "binomial")
benchmark_aggregated_binomial <- aggregate_benchmarks(benchmark_binomial)

# Multinomial -------------------------------------------------------------

dna <- getData("dna", scaled = TRUE)
vehicle <- getData("vehicle", scaled = TRUE)
poker <- getData("poker", "training", scaled = FALSE)
poker$x <- scale(poker$x)

multinomial_datasets <- list(vehicle = vehicle,
                             dna = dna,
                             poker = poker)

benchmark_multinomial <- benchmark(multinomial_datasets, "multinomial")
benchmark_aggregated_multinomial <- aggregate_benchmarks(benchmark_multinomial)

benchmarks <- list(gaussian = benchmark_aggregated_gaussian,
                   binomial = benchmark_aggregated_binomial,
                   multinomial = benchmark_aggregated_multinomial)

usethis::use_data(benchmarks, overwrite = TRUE)

# Multivariate gaussian ---------------------------------------------------

# Violent crimes

tmp_file <- tempfile()
tmp_dir <- tempdir()

download.file(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00211/CommViolPredUnnormalizedData.txt",
  tmp_file
)

d <- read.csv(tmp_file, na.string = "?", header = FALSE)

y1 <- d[, 130:147]
keep <- apply(y1, 1, function(x) all(!is.na(x)))

y1 <- y1[keep, ]
x1 <- d[keep, c(6:30, 32:103, 121:123)]

violence <- list(x = x1, y = y1)

unlink(tmp_file)

# Condition Based Maintenance of Naval Propulsion Plants Data Set
tmp_file <- tempfile()
tmp_dir <- tempdir()

download.file(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
  tmp_file
)

unzip(tmp_file, exdir = tmp_dir)

d <- read.table(file.path(tmp_dir, "UCI CBM Dataset", "data.txt"),
                header = FALSE)

x1 <- d[, -c(9, 17, 18)]
x2 <- scale(x1)
y <- d[, 17:18]

naval <- list(x = x2, y = y)

# Bike sharing
tmp_file <- tempfile()
tmp_dir <- tempdir()

download.file(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
  tmp_file
)

unzip(tmp_file, exdir = tmp_dir)

d <- read.csv(file.path(tmp_dir, "day.csv"))
x1 <- subset(d,
             select = c("season",
                        "yr",
                        "mnth",
                        "holiday",
                        "weekday",
                        "workingday",
                        "weathersit",
                        "temp",
                        "atemp",
                        "hum",
                        "windspeed"))
x2 <- x1
x2[, 1:7] <- sapply(x2[, 1:7], as.factor)
x3 <- model.matrix(~ ., data = x2)
x4 <- Matrix::Matrix(x3[, -1])

y <- as.matrix(subset(d, select = c("casual", "registered")))

bikes <- list(x = x4, y = y)

mgaussian_datasets <- list(violence = violence,
                           bikes = bikes,
                           naval = naval)

benchmark_mgaussian <- benchmark(mgaussian_datasets, "mgaussian")
benchmark_aggregated_mgaussian <- aggregate_benchmarks(benchmark_mgaussian)

benchmarks <- list(gaussian = benchmark_aggregated_gaussian,
                   binomial = benchmark_aggregated_binomial,
                   multinomial = benchmark_aggregated_multinomial,
                   mgaussian = benchmark_aggregated_mgaussian)

usethis::use_data(benchmarks, overwrite = TRUE)
