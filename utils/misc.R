

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
  run_dir <- as_run_dir(run_dir)
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
  keras::load_model_weights_hdf5(model, "model-weights-best-checkpoint.hdf5")
  
  if (return_flags)
    list(model = model, FLAGS = FLAGS)
  else
    model
}
