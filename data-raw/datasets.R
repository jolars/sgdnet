# houses (gaussian) -------------------------------------------------------

library(Matrix)

temp_file <- tempfile()

download.file(
  "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/cadata",
  temp_file
)

tmp <- e1071::read.matrix.csr(temp_file, fac = FALSE)
tmp_x <- as.data.frame(as.matrix(tmp$x))
colnames(tmp_x) <- c("median_income",
                     "housing_median_age",
                     "total_rooms",
                     "total_bedrooms",
                     "population",
                     "households",
                     "latitude",
                     "longitude")

houses <- list(x = tmp_x, y = tmp$y)

usethis::use_data(houses, overwrite = TRUE)

# mushrooms (binomial) ----------------------------------------------------

temp_file <- tempfile()

download.file(
  "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
  temp_file
)

library(readr)
library(forcats)

temp_data <- readr::read_csv(temp_file,
                             col_names = c("edibility",
                                           "cap_shape",
                                           "cap_surface",
                                           "cap_color",
                                           "bruises",
                                           "odor",
                                           "gill_attachment",
                                           "gill_spacing",
                                           "gill_size",
                                           "gill_color",
                                           "stalk_shape",
                                           "stalk_root",
                                           "stalk_surface_above_ring",
                                           "stalk_surface_below_ring",
                                           "stalk_color_above_ring",
                                           "stalk_color_below_ring",
                                           "veil_type",
                                           "veil_color",
                                           "ring_number",
                                           "ring_type",
                                           "spore_print_color",
                                           "population",
                                           "habitat"))

temp_data2 <- temp_data %>%
  # we drop stalk_root which has missing data
  select(-stalk_root) %>%
  # recode into factor variables
  mutate(edibility = fct_recode(edibility,
                                poisonous = "p",
                                edible = "e"),
         cap_shape = fct_recode(cap_shape,
                                bell = "b",
                                conical = "c",
                                convex = "x",
                                flat = "f",
                                knobbed = "k",
                                sunken = "s"),
         cap_surface = fct_recode(cap_surface,
                                  fibrous = "f",
                                  grooves = "g",
                                  scaly = "y",
                                  smooth = "s"),
         cap_color = fct_recode(cap_color,
                                brown = "n",
                                buff = "b",
                                cinnamon = "c",
                                gray = "g",
                                green = "r",
                                pink = "p",
                                purple = "u",
                                red = "e",
                                white = "w",
                                yellow = "y"),
         bruises = fct_recode(bruises,
                              bruises = "t",
                              no = "f"),
         odor = fct_recode(odor,
                           almond = "a",
                           anise = "l",
                           creosote = "c",
                           fishy = "y",
                           foul = "f",
                           musty = "m",
                           none = "n",
                           pungent = "p",
                           spicy = "s"),
         gill_attachment = fct_recode(gill_attachment,
                                      attached = "a",
                                     #descending = "d",
                                     #notched = "n",
                                      free = "f"),
         gill_spacing = fct_recode(gill_spacing,
                                   close = "c",
                                  #distant = "d",
                                   crowded = "w"),
         gill_size = fct_recode(gill_size,
                                broad = "b",
                                narrow = "n"),
         gill_color = fct_recode(gill_color,
                                 black = "k",
                                 brown = "n",
                                 buff = "b",
                                 chocolate = "h",
                                 gray = "g",
                                 green = "r",
                                 orange = "o",
                                 pink = "p",
                                 purple = "u",
                                 red = "e",
                                 white = "w",
                                 yellow = "y"),
         stalk_shape = fct_recode(stalk_shape,
                                  enlarging = "e",
                                  tapering = "t"),
         stalk_surface_above_ring = fct_recode(stalk_surface_above_ring,
                                               fibrous = "f",
                                               scaly = "y",
                                               silky = "k",
                                               smooth = "s"),
         stalk_surface_below_ring = fct_recode(stalk_surface_below_ring,
                                               fibrous = "f",
                                               scaly = "y",
                                               silky = "k",
                                               smooth = "s"),
         stalk_color_above_ring = fct_recode(stalk_color_above_ring,
                                             brown = "n",
                                             buff = "b",
                                             cinnamon = "c",
                                             gray = "g",
                                             orange = "o",
                                             pink = "p",
                                             red = "e",
                                             white = "w",
                                             yellow = "y"),
         stalk_color_below_ring = fct_recode(stalk_color_below_ring,
                                             brown = "n",
                                             buff = "b",
                                             cinnamon = "c",
                                             gray = "g",
                                             orange = "o",
                                             pink = "p",
                                             red = "e",
                                             white = "w",
                                             yellow = "y"),
         veil_type = fct_recode(veil_type,
                               #universal = "u",
                                partial = "p"),
         veil_color = fct_recode(veil_color,
                                 brown = "n",
                                 orange = "o",
                                 white = "w",
                                 yellow = "y"),
         ring_number = fct_recode(ring_number,
                                  none = "n",
                                  one = "o",
                                  two = "t"),
         ring_type = fct_recode(ring_type,
                               #cobwebby = "c",
                                evanescent = "e",
                                flaring = "f",
                                large = "l",
                                none = "n",
                               #sheathing = "s",
                               #zone = "z"
                                pendant = "p"),
         spore_print_color = fct_recode(spore_print_color,
                                        black = "k",
                                        brown = "n",
                                        buff = "b",
                                        chocolate = "h",
                                        green = "r",
                                        orange = "o",
                                        purple = "u",
                                        white = "w",
                                        yellow = "y"),
         population = fct_recode(population,
                                 abundant = "a",
                                 clustered = "c",
                                 numerous = "n",
                                 scattered = "s",
                                 several = "v",
                                 solitary = "y"),
         habitat = fct_recode(habitat,
                              grasses = "g",
                              leaves = "l",
                              meadows = "m",
                              paths = "p",
                              urban = "u",
                              waste = "w",
                              woods = "d"))

x <- Matrix::Matrix(as.matrix(dummy_columns(temp_data2[, -1])[, 22:133]))
y <- pull(temp_data2, edibility)

mushrooms <- list(x = x, y = y)

devtools::use_data(mushrooms, overwrite = TRUE)

unlink(temp_file)

# pendigits (multinomial) -------------------------------------------------

temp_file1 <- tempfile()
temp_file2 <- tempfile()

download.file(
  "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits",
  temp_file1
)

download.file(
  "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t",
  temp_file2
)

test <- e1071::read.matrix.csr(temp_file1, fac = TRUE)
train <- e1071::read.matrix.csr(temp_file2, fac = TRUE)

library(Matrix)

test_x <- as(test$x, "dgCMatrix")
train_x <- as(train$x, "dgCMatrix")

pendigits <- list(test = list(x = test_x, y = test$y),
                  train = list(x = train_x, y = train$y))

usethis::use_data(pendigits, overwrite = TRUE)

unlink(temp_file1)
unlink(temp_file2)
