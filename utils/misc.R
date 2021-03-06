

is_tensor <- function(x) inherits(x, "tensorflow.tensor")


as_tensor <- function(x, dtype = NULL, coerce_to_mat = FALSE) {
  if (is.null(x))
    return(x)
  
  if (coerce_to_mat)
    x %<>% as.matrix()
  
  if (is_tensor(x) && !is.null(dtype))
    tf$cast(x, dtype = dtype)
  else {
    if (rlang::is_integerish(x) && isTRUE(dtype$is_integer))
      storage.mode(x) <- "integer"
    tf$convert_to_tensor(x, dtype = dtype)
  }
}



layer_expand_dims <- function(object, axis = -1L, ...) {
  layer_lambda(object, function(x) {
    tf$expand_dims(x, as.integer(axis))
  }, name = "expand_dims")
}


build_and_compile <- 
  function(input,
           output,
           optimizer = 'adam',
           loss = "categorical_crossentropy",
           metric = 'acc') {
    model <- keras::keras_model(input, output) %>%
      keras::compile(optimizer = optimizer,
                     loss = loss,
                     metric = metric)
    model
  }



restore_model <- function(run_dir, model_name = "model.R", return_flags = FALSE) {
  run_dir <- tfruns::as_run_dir(run_dir)
  owd <- setwd(run_dir)
  on.exit(setwd(owd), add = TRUE)
  if (exists("FLAGS")) {
    envs <- list()
    caches <- list()
    i <- 0
    
    while (exists("FLAGS")) {
      add(i) <- 1
      envs[[i]] <- pryr::where("FLAGS")
      caches[[i]] <- FLAGS
      rm(FLAGS, envir = envirs[[i]])
    }
    
    on.exit(lapply(seq_len(i), function(ii) {
      assign("FLAGS", caches[[i]], envir = envs[[i]])
    }), add = TRUE)
  }
  
  env.utils::import_from(model_name, model, FLAGS)
  keras::load_model_weights_hdf5(model, "model-weights-best-checkpoint.h5")
  
  if (return_flags)
    list(model = model, FLAGS = FLAGS)
  else
    model
}

# For predicting with 3 softmax outputs (168, 11, 7), 12 == number of test examples
argmax <- 
  function(x) { lapply(x, function(y) lapply(1:12, function(i) y[i,] %>% which.max())) }


timestamp <- function() lubridate::now() %>% stringr::str_replace(" ", "_")


is_scalar <- function(x) identical(length(x), 1L)

# resblock with bottleneck from the Exploring Normalization FB paper (01/2017)
# modified as in: https://www.kaggle.com/h030162/version1-0-9696
resblock_batchnorm_bottle_2d <-
  function(x, filters = 64L, downsample = TRUE, renorm = FALSE, kernel_size = 3L) {
    a <- x %>%
      layer_conv_2d(filters, 1, padding = 'same') %>%
      layer_batch_normalization() %>%
      layer_activation_relu()
    
    b <- a %>%
      layer_conv_2d(filters, kernel_size, padding = 'same') %>%
      layer_batch_normalization(renorm = renorm) %>%
      layer_activation_relu()
    
    c <- b %>%
      layer_conv_2d(filters, kernel_size, padding = 'same') %>%
      layer_batch_normalization(renorm = renorm)
    
    shape <- x$shape$as_list()
    og_filters <- shape[[length(shape)]]
    
    
    residual <-
      if (downsample) {
        residual <- layer_conv_2d(x, og_filters, 1, padding = 'same')
      } else x
    
    se_out <- c %>% 
      se_module(2L) %>% 
      layer_conv_2d(og_filters, 1, padding = 'same')
    
    out <- layer_add(list(se_out, residual))
    
    out
  }
