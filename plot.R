rundir <- normalizePath("~/internal/runs/2020-02-22T13-49-35.272Z")

histfile <- file.path(rundir, "history.qs")
hist <- qs::qread(histfile)

p <- plot(hist)

