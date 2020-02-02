library(reticulate)

microbenchmark::microbenchmark({
  source_python("python/import_pd.py")
}, times = 1L) %>% print()


microbenchmark::microbenchmark({
  source_python("python/import_pq.py")
}, times = 1L) %>% print()


microbenchmark::microbenchmark({
  source_python("python/import_npa.py")
}, times = 1L) %>% print()


