#!/usr/bin/Rscript

reticulate::source_python("python/load_test_data.py")


# import_from("python/tfrecords/img_generator.py", "test_generator")
# TODO: add docopt to scripts
# TODO: Restore model and predict on test data
# TODO: preprocess test image.  e.g. crop/resize/normalize
# TODO: Format outputs like sample_submission.csv

if (length(args) < 2)
  stop("Usage: eval.R RUNDIR CSVFILE")

args    <- commandArgs(TRUE)
rundir  <- normalizePath(file.path("~/internal/runs/", args[[1]]))
csvfile <- normalizePath(args[[2]])


model <- restore_model(rundir)


npa <- load_test_data()


predictions <- vector("list", 12L*3L)

i <- 1L
j <- 1L
for (img in npa) {
  browser()
  pred <- img %>% 
    model$predict() %>% 
    lapply(function(x) tf$argmax(x))
  
  names(pred) <- c("root", "cons", "vowel")
  
  row_id.root  <- paste0("Test_", j-1, "grapheme_root")
  row_id.cons  <- paste0("Test_", j-1, "consonant_diacritic")
  row_id.vowel <- paste0("Test_", j-1, "vowel_diacritic")
  
  print(pred)
  
  predictions[[i]]   <- c(row_id.conv,  pred$cons)
  predictions[[i+1]] <- c(row_id.root,  pred$root)
  predictions[[i+2]] <- c(row_id.vowel, pred$vowel)
  i <- i + 3L
  j <- j + 1L
}

str(predictions)
preds <- tibble::as_tibble(predictions)


write_csv(preds, csvfile)

