
source_python("python/read_tfrecord_dataset.py")


ds <- ds_parsed %>%
  dataset_map(function(x) {
    x$image_raw$set_shape(list(HEIGHT, WIDTH, 1L))
    tuple(x$image_raw,
          tuple(x$label_grapheme, x$label_consonant, x$label_vowel))
  }) %>%
  dataset_shuffle(100L) %>% 
  dataset_batch(128L) %>%
  dataset_prefetch(10L)

val_ds <- ds$take(100L)
