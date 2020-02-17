library(reticulate)
library(keras)
library(tensorflow)
library(tfdatasets)
library(magrittr)
library(zeallot)
library(readr)
library(tibble)
library(env.utils)
suppressPackageStartupMessages(library(dplyr))
# suppressPackageStartupMessages(library(deepR))
# suppressPackageStartupMessages(library(JSGutils))

# TODO: source utils folder like Tomasz
source("utils/labels.R")
source("utils/misc.R")
source_python("python/data_tools.py")

# test data
X <- tf$random$normal(shape = list(8L, 256L, 3L, 16L))
x <- tf$random$normal(shape = list(8L, 256L, 16L))
