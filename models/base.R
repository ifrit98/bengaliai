
HEIGHT <- 64
WIDTH  <- 64

input <- layer_input(shape = list(HEIGHT, WIDTH, 1L))

base <- input %>% 
  layer_conv_2d(16L, 3L, activation = 'relu') %>% 
  layer_conv_2d(32L, 5L, activation = 'relu') %>% 
  layer_batch_normalization() %>% 
  layer_conv_2d(64L, 10L, activation = 'relu') %>% 
  layer_conv_2d(96L, 12L, activation = 'relu') %>% 
  layer_batch_normalization() %>% 
  layer_global_max_pooling_2d()


root <- base %>% 
  layer_dense(192L) %>% 
  layer_dense(168L, activation = 'softmax')

cons <- base %>% 
  layer_dense(64L) %>% 
  layer_dense(7L, activation = 'softmax')

vowel <- base %>% 
  layer_dense(64L) %>% 
  layer_dense(11L, activation = 'softmax')


model <- keras_model(input, list(root, cons, vowel))

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = 'acc'
)

cat("Finished sourcing model\n")
