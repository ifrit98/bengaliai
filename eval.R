#!/usr/bin/Rscript

argmax <- 
  function(x) { lapply(x, function(y) lapply(1:32, function(i) y[i,] %>% which.max())) }

reticulate::source_python("python/load_test_data.py")

rundir <- ("./")


model <- restore_model(rundir)

import_from("dataset.R", val_ds)
  

sess <- tf$compat$v1$keras$backend$get_session()



predict <- function(input) {
  raw_probs <- sess$run(model(input))
  
  preds <- raw_probs %>% 
    argmax() %>% 
    listarrays::bind_as_cols() %>% 
    tibble::as_tibble()

  names(preds) <- c("grapheme_root", "consonant_diacritic", "vowel_diacritic")
  
  tidyr::unnest(preds, cols = names(preds))
}



evaluate <- function() {
  
  nb <- next_batch(val_ds)
  batch  <- sess$run(nb)
  input  <- batch[[1]]
  labels <- batch[[2]]
  
  preds <- input %>% 
    as_tensor(tf$float32) %>% 
    predict() %>% 
    tibble::rowid_to_column(var = "rowid")
  
  
  labs <- vector("list", 32)
  
  for (i in seq(32)) {
    labs[[i]] <- lapply(labels, function(x) which.max(x[i,]))
    names(labs[[i]]) <- c("grapheme_root", "consonant_diacritic", "vowel_diacritic")
  }
  
  truth <- labs %>% 
    listarrays::bind_as_rows() %>% 
    tibble::as_tibble() %>% 
    tidyr::unnest(cols = names(.))
  
  
  preds$rowid <- NULL
  logits <- (preds == truth) %>% tibble::as_tibble()
  
  
  grapheme_acc  <- logits %>% { sum(.$grapheme_root) / length(.$grapheme_root) }
  consonant_acc <- logits %>% { sum(.$consonant_diacritic) / length(.$consonant_diacritic) }
  vowel_acc     <- logits %>% { sum(.$vowel_diacritic) / length(.$vowel_diacritic) }
  
  out <- c(grapheme_acc, consonant_acc, vowel_acc)
  names(out) <- c("grapheme_acc", "consonant_acc", "vowel_acc")
  
  out
}



for (i in seq(3)) {
  accuracy <- evaluate()
  message("Success!")
  cat("Batch ", i, "Grapheme accuracy:",  accuracy[[1]], "\n")
  cat("Batch ", i, "Consonant accuracy:", accuracy[[2]], "\n")
  cat("Batch ", i, "Vowel accuracy:",     accuracy[[3]], "\n")
}

