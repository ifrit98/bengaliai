# rundir <- normalizePath("~/internal/runs/2020-02-22T13-49-35.272Z")


histfile <- "history.qs" #file.path(rundir, "history.qs")
hist <- qs::qread(histfile)

outfile <- "history.png" #file.path(rundir, "history.png")

png(filename = outfile)
plot(hist)
dev.off()

png(filename = "plots/")