library(JSGutils)
library(tensorflow)
library(reticulate)
library(magrittr)

source_python("data-raw/import_pq.py")

# df_train0 %<>% JSGutils::drop_col(image_id)
df_test0  %<>% drop_col(image_id)


image_tensors <-
  lapply(1:length(df_test0[, 1]),
         function(i) {
           unname(unlist(df_test0[i,][-1])) %>% as_tensor(dtype = tf$float32)
         })



image_tensors_2d <- 
  lapply(image_tensors, function(x) tf$reshape(x, shape = list(137L, 236L)))


# TODO: Figure out:
#    batching
#    efficient storage (tfrecords?)
#    preprocessing, impairments, and loading into model 
#    drop low value pixels < ~28?