library(JSGutils)
library(tensorflow)
library(reticulate)
library(magrittr)

source_python("data-raw/import_pq.py")

# df_train0 %<>% JSGutils::drop_col(image_id)
df_test0  %<>% drop_col(image_id)

images <- images0

image_tensors <-
  lapply(1:(dim(images)[1]),
         function(i) {
           images[i,,] %>% JSGutils::as_tensor()
         }) %>% tf$stack()


# TODO: Figure out:
#    batching
#    efficient storage (tfrecords?)
#    preprocessing, impairments, and loading into model 
#    drop low value pixels < ~28?