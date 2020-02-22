#!/usr/bin/Rscript
# TODO: add docopt to scripts

reticulate::source_python("python/load_test_data.py")

# if (length(args) < 2)
#   stop("Usage: eval.R RUNDIR CSVFILE")
# 
# args    <- commandArgs(TRUE)
# rundir  <- normalizePath(file.path("~/internal/runs/", args[[1]]))
# csvfile <- normalizePath(args[[2]])

rundir <- normalizePath("~/internal/runs/2020-02-22T13-49-35.272Z")
# rundir <- normalizePath("~/internal/runs/2020-02-17T18-53-46.997Z")
csvfile <- paste0("submissions/", timestamp(), ".csv")

model <- restore_model(rundir)


# testfiles <- list('data/data-raw/test_image_data_0.parquet')
npa <- load_test_data()


sess <- tf$compat$v1$keras$backend$get_session()


predict <- function(input) {
  input <- input %>% 
    as_tensor(dtype = tf$float32) %>% 
    layer_expand_dims()
  
  raw_probs <- sess$run(model(input)) 
  # browser()
  
  preds <- raw_probs %>% 
    argmax() %>% 
    listarrays::bind_as_cols() %>% 
    tibble::as_tibble()

  names(preds) <- c("grapheme_root", "consonant_diacritic", "vowel_diacritic")
  
  tidyr::unnest(preds, cols = names(preds))
}


preds <- npa %>% 
  predict() %>% 
  tibble::rowid_to_column(var = "rowid")


predictions <- lapply(purrr::transpose(preds), function(pred) {
  idx <- pred$rowid - 1L
  row_id.root  <- paste0("Test_", idx, "_grapheme_root")
  row_id.cons  <- paste0("Test_", idx, "_consonant_diacritic")
  row_id.vowel <- paste0("Test_", idx, "_vowel_diacritic")
  
  cons <- c(row_id.cons, pred$consonant_diacritic)
  root <- c(row_id.root, pred$grapheme_root)
  vow  <- c(row_id.vowel, pred$vowel_diacritic)

  
  predictions <- list(cons, root, vow) %>%  
    listarrays::bind_as_rows() %>% 
    tibble::as_tibble()
  
  names(predictions) <- c("row_id", "target")
  
  predictions
})

str(predictions)
csv_preds <- dplyr::bind_rows(predictions)
csv_preds$target <- as.double(csv_preds$target)


readr::write_csv(csv_preds, csvfile)

message("Success!")
