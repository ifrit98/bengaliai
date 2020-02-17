

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