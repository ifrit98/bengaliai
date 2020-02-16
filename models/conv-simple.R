if (!exists_here("FLAGS")) {
  import_from("flags.R", FLAGS)
}

input <- layer_input(shape = list(FLAGS$height, FLAGS$width, 1L))

base <- input %>% 
  layer_conv_2d(32L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(32L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(32L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(32L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_batch_normalization(momentum = 0.9) %>% 
  layer_max_pooling_2d(pool_size = list(2L, 2L)) %>% 
  layer_conv_2d(32L, 5L, activation = 'relu', padding = 'same') %>% 
  layer_dropout(rate = 0.3)

block1 <- base %>% 
  layer_conv_2d(64L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(64L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(64L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(64L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_batch_normalization(momentum = 0.9) %>% 
  layer_max_pooling_2d(pool_size = list(2L, 2L)) %>% 
  layer_dropout(rate = 0.3)

features <- block1 %>% 
  layer_conv_2d(128L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(128L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(256L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(512L, 3L, activation = 'relu', padding = 'same') %>% 
  layer_batch_normalization(momentum = 0.9) %>% 
  layer_max_pooling_2d(pool_size = list(2L, 2L)) %>% 
  layer_dropout(rate = 0.3)


features_flat <- 
  if (!FLAGS$global_pool) layer_flatten(features) else layer_global_max_pooling_2d(features) 


features_dense <- features_flat %>%
  layer_dense(256, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(192, activation = 'relu')


root <- features_dense %>% 
  # layer_dense(192) %>% 
  layer_dense(length(GPH$index), activation = 'softmax', name = "grapheme_root")

cons <- features_dense %>% 
  # layer_dense(64L) %>% 
  layer_dense(length(CON$index), activation = 'softmax', name = "consonant")

vowel <- features_dense %>% 
  # layer_dense(64L) %>% 
  layer_dense(length(VOW$index), activation = 'softmax', name = "vowel")



model <- keras_model(input, list(root, cons, vowel))

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = 'acc'
)

cat("Finished sourcing model with %s params\n", model$count_params())
