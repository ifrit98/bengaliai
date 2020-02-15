if (!exists_here("FLAGS")) {
  import_from("flags.R", FLAGS)
}

input <- layer_input(shape = list(FLAGS$height, FLAGS$width, 1L))

base <- input %>% 
  layer_conv_2d(16L, 3L, activation = 'relu') %>% 
  layer_conv_2d(32L, 5L, activation = 'relu') %>% 
  layer_batch_normalization() %>% 
  layer_conv_2d(48L, 8L, activation = 'relu') %>% 
  layer_conv_2d(64L, 10L, activation = 'relu') %>% 
  layer_batch_normalization() %>% 
  layer_global_max_pooling_2d()


root <- base %>% 
  layer_dense(192L) %>% 
  layer_dense(length(GPH$index), activation = 'softmax', name = "grapheme_root")

cons <- base %>% 
  layer_dense(64L) %>% 
  layer_dense(length(CON$index), activation = 'softmax', name = "consonant")

vowel <- base %>% 
  layer_dense(64L) %>% 
  layer_dense(length(VOW$index), activation = 'softmax', name = "vowel")


model <- keras_model(input, list(root, cons, vowel))

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = 'acc'
)

cat("Finished sourcing model with %s params\n", model$count_params())
