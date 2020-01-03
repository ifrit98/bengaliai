
library(magrittr)
library(zeallot)
library(readr)
train <- read_csv("data-raw/train.csv")
test  <- read_csv("data-raw/test.csv")
map   <- read_csv("data-raw/class_map.csv")
sub   <- read_csv("data-raw/sample_submission.csv")


# Loads 1st train image .parquet file as pandas data 
reticulate::source_python("data-raw/import_pq.py")

# df_train <- df_train0 %>% JSGutils::drop_col(image_id)
df_test <- df_test0 %>% JSGutils::drop_col(image_id)



slice <- function(x, window_len = 137) {
  x <- as.vector(x)
  ncol2(x, pad = NULL) <- window_len
  
  # Drop remainder if LT window size
  if (length(x[dim(x)[1],]) < window_len)
    x <- x[-dim(x)[1],]
  
  lapply(seq_len(nrow(x)), function(i) x[i,])
}


# parallel::mcMap ?
# parallel::mclapply ?



unflatten_images <- function(df, dims = c(137, 236, 1)) {
  purrr::map(1:length(df[, 1]),
             function(i) {
               x <- df[i, ]
               listarrays::dim2(x) <- dims
               x
             })
}


df <- unflatten_images(df_test)