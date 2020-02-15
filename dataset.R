
source_python("python/tfrecords/load_tfrecords.py")
# import_from('python/tfrecords/tfrecord_utils.py', replay_generator)

if (!exists_here("FLAGS")) {
  import_from("flags.R", FLAGS)
}

# ds_raw <- replay_generator(FLAGS$source_dir)
# (b <- next_batch(ds_raw))
# sess <- tf$Session()
# bb <- sess$run(b)

ds <- ds_raw %>%
  dataset_map(function(x) {
    # browser()
    x$image     <- tf$expand_dims(x$image, -1L) %>% tf$squeeze(0L)
    x$grapheme  <- tf$one_hot(x$grapheme, length(GPH$index), dtype = tf$int32) %>% tf$squeeze(0L)
    x$vowel     <- tf$one_hot(x$vowel, length(VOW$index), dtype = tf$int32) %>% tf$squeeze(0L)
    x$consonant <- tf$one_hot(x$consonant, length(CON$index), dtype = tf$int32) %>% tf$squeeze(0L)
    
    tuple(x$image,
          tuple(x$grapheme, x$consonant, x$vowel))
  }) %>%
  dataset_batch(FLAGS$batch_size, drop_remainder = TRUE) %>%
  dataset_shuffle(100L) %>% 
  dataset_repeat() %>%
  dataset_prefetch(2L)
  
# nb <- next_batch(ds)
# nbb <- sess$run(nb)
# nbb[[2]]
  
val_ds <- ds$take(FLAGS$val_size)

# TODO: Do dataset preprocessing here?
# TODO: auxmix, scaling, and normalization