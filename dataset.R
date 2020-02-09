
source_python("python/tfrecords/load_tfrecords.py")
# import_from('python/tfrecords/tfrecord_utils.py', replay_generator)

if (!exists_here("FLAGS")) {
  import_from("flags.R", FLAGS)
}

# ds_raw <- replay_generator(FLAGS$source_dir)


(b <- next_batch(ds_raw))

ds <- ds_raw %>%
  dataset_map(function(x) {
    x$image <- tf$reshape(x$image, list(137L, 236L, 1L))
    tuple(x$image,
          tuple(x$grapheme, x$consonant, x$vowel))
  }) %>%
  dataset_shuffle(10L) %>% 
  dataset_batch(FLAGS$batch_size) %>%
  dataset_prefetch(2L)

val_ds <- ds$take(FLAGS$val_size)

(b <- next_batch(ds))

# TODO: Do dataset preprocessing here?
# TODO: auxmix, scaling, and normalization