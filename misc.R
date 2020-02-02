

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
