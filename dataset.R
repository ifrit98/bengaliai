
source_python("python/read_tfrecord_dataset.py")


ds <- ds_parsed %>%
  dataset_map(function(x) {
    x$image_raw$set_shape(list(HEIGHT, WIDTH, 1L))
    tuple(x$image_raw,
          tuple(x$label_grapheme, x$label_consonant, x$label_vowel))
  }) %>%
  dataset_shuffle(10L) %>% 
  dataset_batch(FLAGS$batch_size) %>%
  dataset_prefetch(2L)

val_ds <- ds$take(FLAGS$val_size)


# TODO: Do dataset preprocessing here?
# TODO: auxmix, scaling, and normalization