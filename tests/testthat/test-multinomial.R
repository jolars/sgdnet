x <- as.matrix(iris[, 1:4])
y <- iris[, 5]

# TODO(jolars): add more tests here

test_that("predicted responses sum to 1", {
  sfit <- sgdnet(x, y, alpha = 0, family = "multinomial")

  rowsums <- apply(predict(sfit, x, type = "response"), c(1, 3), sum)
  expect_equivalent(rowsums, matrix(1, nrow(rowsums), ncol(rowsums)))
})

