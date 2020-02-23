if (!exists_here("FLAGS")) {
  import_from("flags.R", FLAGS)
}

se_module <- function(x, filters, reduction_factor = 2L) {
  
  reduced <- filters %/% reduction_factor
  
  residual <- x
  
  x <- x %>% 
    layer_average_pooling_2d(1) %>% 
    layer_conv_2d(reduced, 1, padding = 'same') %>% 
    layer_activation_relu() %>% 
    layer_conv_2d(filters, 1, padding = 'same') %>% 
    layer_activation(activation = 'sigmoid')
  
  layer_multiply(list(x, residual))
}


make_layer <-
  function(x,
           block,
           filters,
           no_blocks,
           reduction_factor,
           expansion,
           stride = 1) {
    shape <- x$shape$as_list()
    downsample <- NULL
    inchannels <- shape[[length(shape)]]
    
    if (stride != 1L || inchannels != filters * expansion) {
      x <- x %>% 
        layer_conv_2d(filters * expansion, 1, padding = 'same', use_bias = FALSE) %>% 
        layer_batch_normalization()
    }
    
    x <- block(x, filters, reduction_factor, expansion = expansion)
    
    for (i in seq(2, no_blocks)) 
      x <- block(x, filters, reduction_factor, expansion = expansion) 
    
    x
}


# inplanes, planes, groups, reduction, stride, downsampled
# https://www.kaggle.com/h030162/version1-0-9696
SEResNeXtBottleneck <-
  function(x,
           filters,
           reduction = 16L,
           base_width = 4L,
           stride = 1,
           expansion = 2L,
           downsample = NULL) {
    
    width <- floor(filters * (base_width / 64L)) * 32L # groups = 32
    # browser()
    
    residual <- x
    
    conv1 <- x %>% 
      layer_separable_conv_2d(width, 3, padding = 'same', strides = stride, depth_multiplier = 1) %>% 
      # layer_depthwise_conv_2d(3, padding = 'same', strides = stride, depth_multiplier = 2) %>% 
      # layer_conv_2d(width, 1, use_bias = FALSE, padding = 'same', strides = 1) %>% 
      layer_batch_normalization() %>% 
      layer_activation_relu()
    
    conv2 <- conv1 %>% 
      layer_separable_conv_2d(width, 3, padding = 'same', strides = stride, depth_multiplier = 1) %>% 
      # layer_depthwise_conv_2d(3, padding = 'same', strides = stride, depth_multiplier = 2) %>% 
      # layer_conv_2d(width, 3, padding = 'same', strides = stride) %>% # depthwise separable convolution?
      layer_batch_normalization() %>% 
      layer_activation_relu()
    
    conv3 <- conv2 %>% 
      layer_conv_2d(filters * expansion, 1, use_bias = FALSE, padding = 'same') %>% 
      layer_batch_normalization()
    
    se_out <- se_module(conv3, filters * expansion, reduction_factor = reduction)
    
    out <- layer_add(list(se_out, residual))
  
    out  
}


# num_blocks -> layers = [3, 4, 6, 3]
# groups = 32
# reduction = 16
# inplanes = 64
# https://arxiv.org/abs/1709.01507
SE_Net <-
  function(num_classes = 186L, # 168 + 11 + 7
           filters = list(32L, 64L, 128L, 256L),
           expansion = 2L,
           no_blocks = 1L,
           reduction = 16L) {
    
  if (is_scalar(no_blocks)) no_blocks %<>% rep(4L)
  # browser()
    
  input <- layer_input(shape = list(128L, 128L, 1L))
  
  layer0 <- input %>% 
    layer_conv_2d(filters[[1]], 3, strides = 2, padding = 'same', use_bias = FALSE) %>% 
    layer_batch_normalization() %>% 
    layer_activation_relu() %>% 
    layer_conv_2d(filters[[1]], 3, strides = 1, padding = 'same', use_bias = FALSE) %>% 
    layer_batch_normalization() %>% 
    layer_activation_relu() %>% 
    layer_conv_2d(filters[[1]], 3, strides = 1, padding = 'same', use_bias = FALSE) %>% 
    layer_batch_normalization() %>% 
    layer_activation_relu() %>% 
    layer_max_pooling_2d(pool_size = 3, strides = 2)
  
  
  layer1 <- layer0 %>% 
    make_layer(
      block = SEResNeXtBottleneck,
      filters = filters[[1]],
      no_blocks = no_blocks[[1]],
      reduction = reduction,
      expansion = expansion
    )
  
  layer2 <- layer1 %>% 
    make_layer(
      block = SEResNeXtBottleneck,
      filters = filters[[2]],
      no_blocks = no_blocks[[2]],
      reduction = reduction,
      expansion = expansion
    )
  
  layer3 <- layer2 %>% 
    make_layer(
      block = SEResNeXtBottleneck,
      filters = filters[[3]],
      no_blocks = no_blocks[[3]],
      reduction = reduction,
      expansion = expansion
    )
  
  layer4 <- layer3 %>% 
    make_layer(
      block = SEResNeXtBottleneck,
      filters = filters[[4]],
      no_blocks = no_blocks[[4]],
      reduction = reduction,
      expansion = expansion
    )
  
  
  features <- layer4 %>% 
    layer_global_max_pooling_2d() %>% 
    layer_dropout(0.1) 
  
  
  root <- features %>% 
    layer_dense(length(GPH$index), activation = 'softmax', name = "grapheme_root")
  
  cons <- features %>% 
    layer_dense(length(CON$index), activation = 'softmax', name = "consonant")
  
  vowel <- features %>% 
    layer_dense(length(VOW$index), activation = 'softmax', name = "vowel")
  
  
  model <- keras_model(input, list(root, cons, vowel))
  
  model %>% compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metric = 'acc'
  )
  
  model
}

# TODO: try increasing strtide and max pooling down to 1x1 or 8x8


model <- SE_Net(no_blocks = FLAGS$no_blocks)

cat("Number of parameters:", model$count_params(), "\n")

