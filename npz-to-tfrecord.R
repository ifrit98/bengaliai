
source_python("python/data_tools.py")


r0 <- np$load("data-npz/resized_64/train_root_labels0.npz")
r0 <- r0$f[['train_root_labels']]

v0 <- np$load("data-npz/resized_64/train_vowel_labels0.npz")
v0 <- v0$f[['train_vowel_labels']]

c0 <- np$load("data-npz/resized_64/train_consonant_labels0.npz") 
c0 <- c0$f[['train_consonant_labels']]

t0 <- np$load("data-npz/resized_64/train_images0.npz") 
df <- t0$f[['train_images']]

# test0 <- np$load("data-npz/resized_64/test_images0.npz")
# dftest <- 
# 
# rtest0 <- np$load("data-npz/resized_64/test_root_labels0.npz")
# rtest0 <- rtest0$f[['test_root_labels']]
# 
# vtest0 <- np$load("data-npz/resized_64/test_vowel_labels0.npz")
# vtest0 <- vtest0$f[['test_vowel_labels']]
# 
# ctest0 <- np$load("data-npz/resized_64/test_consonant_labels0.npz") 
# ctest0 <- ctest0$f[['test_consonant_labels']]

  


decode_onehot <- function(x) tf$argmax(x)$numpy() %>% as.integer()


train <- as_tensor(df)

rt <- as_tensor(r0)
vt <- as_tensor(v0)
ct <- as_tensor(c0)



serialize_image <- function(image, labels) {
  
  image_str <- img %>% tf$io$serialize_tensor()
  
  raw_image = bytes_feature(image_str)
  label_consonant = int64_feature(labs[[1]])
  label_vowel = int64_feature(labs[[2]])
  label_grapheme = int64_feature(labs[[3]])
  
  feature <- list(
    image = raw_image,
    cons  = label_consonant,
    vowel = label_vowel,
    graph = label_grapheme
  )
  
  proto <- tf$train$Example(features=tf$train$Features(feature=feature))
  
  proto$SerializeToString()
}


# Serialize
serialized_example <- serialize_image(img, labs)


# Serialize whole ds at once
serialized_example_ds <- 


# Get it back
proto1 <- tf$train$Example$FromString(serialized_example)


# Write whole tf$data$Dataset object to a tfrecord file(s) with a tfrecord writer
tmpfile <- tempfile()
# writer <- tf$data$experimental$TFRecordWriter(tmpfile)
# writer$write(serialized_example_ds)


writer <- tf$io$TFRecordWriter(tmpfile)
for (i in 1:5) { # train$shape[[1]]
  img <- train[i,,,]
  labs <- c(rt[i,], vt[i,], ct[i,]) %>% lapply(decode_onehot)
  example <- serialize_image(img, labs)
  writer$write(example)
}

# # iteratively write one example at a time
# with(writer <- tf$io$TFRecordWriter(tmpfile), {
#   for (i in 1:5) { # train$shape[[1]]
#     img <- train[i,,,]
#     labs <- c(rt[i,], vt[i,], ct[i,]) %>% lapply(decode_onehot)
#     example <- serialize_image(img, labs)
#     writer$write(example)
#   }
# })




# Read back in
ds <- tf$data$TFRecordDataset(tmpfile)


# Create a dictionary describing the features
img_feature_desc <- list(
  image = tf$io$FixedLenFeature(list(), tf$string),
  cons  = tf$io$FixedLenFeature(list(), tf$int64),
  vowel = tf$io$FixedLenFeature(list(), tf$int64),
  graph = tf$io$FixedLenFeature(list(), tf$int64)
)



parse_image <- function(proto) {
  ex <- tf$io$parse_single_example(proto, img_feature_desc)
  # Can't change from tf$example tensor before getting batch
  # img <- tf$io$parse_tensor(ex$image, tf$float64)
  # ex$image <- img
  ex
}

# Map over tfrecord dataset and parse images
parsed_img_ds <- ds$map(parse_image)


# Grab a batch to see what's inside
b <- next_batch(parsed_img_ds)

image_raw <- b$image$numpy()

# Convert back to float64 tensor
parsed_tensor <- tf$io$parse_tensor(image_raw, tf$float64)
parsed_tensor %>% print()






