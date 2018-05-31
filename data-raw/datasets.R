# Import permeability dataset ---------------------------------------------

data(permeability, package = "AppliedPredictiveModeling")

permeability <- list(x = Matrix::Matrix(fingerprints),
                     y = permeability)

usethis::use_data(permeability)
