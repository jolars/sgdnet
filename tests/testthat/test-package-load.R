test_that("default options are set when package is loaded", {
  detach("package:sgdnet")
  library("sgdnet")
  expect_equal(getOption("sgdnet.debug"), FALSE)
})
