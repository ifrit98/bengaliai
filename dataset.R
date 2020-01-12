r0 <- np$load("data-npz/train_root_labels0.npz")
r0 <- r0$f[['train_root_labels']]

v0 <- np$load("data-npz/train_vowel_labels0.npz")
v0 <- v0$f[['train_vowel_labels']]

c0 <- np$load("data-npz/train_consonant_labels0.npz") 
c0 <- c0$f[['train_consonant_labels']]

t0 <- np$load("data-npz/train_images0.npz") 
df <- t0$f[['train_images']]



train <- as_tensor(df)

rt <- as_tensor(r0)
vt <- as_tensor(v0)
ct <- as_tensor(c0)


tup <- tuple(train, tuple(r0, c0, v0))
labels <- list(r0, c0, v0)
tlabels <- list(rt, ct, vt)


train_df <- train$numpy()


img <- train[1,,,]
img_str <- img %>% tf$io$serialize_tensor()
labs <- c(rt[1,], vt[1,], ct[1,])
labels <- labs %>% lapply(function(x) as.integer(tf$argmax(x)$numpy()))

# SUCCESS!
source_python("python/serialize_img.py")
serialized_image <- serialize_image(img, labs)

# How to get labels if already in tensors?  R data frames?
serialize_img_wrapper <- function(image, labels) {
  browser()
  img_str <- tf$io$serialize_tensor(image)
  labels %<>% lapply(function(x) as.integer(tf$argmax(x)$numpy()))
  serialize_image(img_str, labels)
}


# TODO: Use raw data when importing from .pq and .csv for features, maybe from R dataframes
source_python("python/load_data.py")
## -- start here -- ##


# TODO: Create tfdataset impairment blocks in R for auxmix.py and others to pipe through here
# TODO: Create tfrecords dataset from tensor slices?
# TODO: Implement Dynamic Routing Capsule network?
# TODO: Evaluation scripts and metrics
# TODO: Implement CLR and LR range test for this dataset
# TODO: implement tf serialize methods/procedures
library(tfdatasets)
tds <- tensor_slices_dataset(tuple(train, tuple(rt, ct, vt))) #%>% 
  dataset_batch(128L, drop_remainder = TRUE) %>% 
  dataset_prefetch(10L)

  
serialized_ds <- tds %>% 
  dataset_map(serialize_img_wrapper)
  
  
  
tdsv <- tds$take(1000L)

ds <- tensor_slices_dataset(train) %>% 
  dataset_batch(32L, drop_remainder = TRUE) %>% 
  dataset_prefetch(2L)


# TODO: serialize in python, will be easier
  dataset_map(map_func = function(x, y) {
    xs <- tf$io$serialize_tensor(x) %>% cast_like(x)
    y1 <- tf$io$serialize_tensor(y[[1]]) %>% cast_like(y[[1]])
    y2 <- tf$io$serialize_tensor(y[[2]]) %>% cast_like(y[[2]])
    y3 <- tf$io$serialize_tensor(y[[3]]) %>% cast_like(y[[3]])
    out <- tuple(xs, y1, y2, y3)# tuple(xs, tuple(y1, y2, y3))
    b()
  })




fn <- tempdir()
writer <- tf$data$experimental$TFRecordWriter(fn)
writer$write(serialized_ds)




library(keras)

if(!exists_here("model"))
  import_from("models/base.R", model) # source("models/base.R")


model %>% 
  fit(tds,
      # validation_data = tdsv,
      epochs = 5L)
  
  fit(model,
      train,
      labels,
      batch_size = 128L,
      epochs = 5L)
    




# source_python("python/import_pd.py")
# t1  <- pd[, -1]
# m   <- t1 %>% as.matrix()
# tbl <- t(t1) %>% as_tibble()
# x <- m %>% as_tensor()
# X <- tf$concat(list(x, x, x, x), axis = 0L)
# X %<>% tf$reshape(shape(-1L, 137L, 236L))
# 
# 
# source_python("python/import_npa.py")
# y <- npa %>% as_tensor()
# Y <- tf$concat(list(y, y, y, y), axis = 0L)
# 
# 
# library(tfdatasets)
# ds <- tensor_slices_dataset(Y)
# 
# ds %<>% {.$batch(1L)} %>% {.$prefetch(1L)}
# 
# b <- next_batch(ds)
# 
# 
# # df_train0 %<>% JSGutils::drop_col(image_id)
# df_test0  %<>% drop_col(image_id)
# 
# images <- images0
# 
# image_tensors <-
#   lapply(1:(dim(images)[1]),
#          function(i) {
#            images[i,,] %>% as_tensor()
#          }) %>% tf$stack()


# TODO: Figure out:
#    batching
#    efficient storage (tfrecords?)
#    preprocessing, impairments, and loading into model 
#    drop low value pixels < ~28?

