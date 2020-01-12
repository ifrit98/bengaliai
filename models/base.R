
library(keras)

input <- layer_input(shape = list(64L, 64L, 1L))

base <- input %>% 
  layer_conv_2d(16L, 3L, activation = 'relu') %>% 
  layer_conv_2d(32L, 5L, activation = 'relu') %>% 
  layer_batch_normalization() %>% 
  layer_conv_2d(64L, 10L, activation = 'relu') %>% 
  layer_conv_2d(128L, 16L, activation = 'relu') %>% 
  layer_batch_normalization() %>% 
  layer_global_max_pooling_2d()


root <- base %>% 
  layer_dense(256L) %>% 
  layer_dense(168L, activation = 'softmax')

cons <- base %>% 
  layer_dense(256L) %>% 
  layer_dense(7L, activation = 'softmax')

vowel <- base %>% 
  layer_dense(256L) %>% 
  layer_dense(11L, activation = 'softmax')


model <- keras_model(input, list(root, cons, vowel))

model %>% compile(
  optimizer = 'adam',
  loss = 'categorical_crossentropy',
  metrics = 'acc'
)
