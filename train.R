
source("flags.R")
# source("dataset.R")
source("dataset-npz.R")

source("models/base.R")


callbacks <- list(
  callback_reduce_lr_on_plateau(monitor = "loss"),
  callback_model_checkpoint("model-weights-best-checkpoint.h5", monitor = "loss")
  # callback_tensorboard(file.path("logs", stringr::str_squish(lubridate::now())))
)


hist <- model %>% 
  fit(
    ds,
    validation_data = val_ds,
    epochs = 100,
    steps_per_epoch = 250,
    callbacks = callbacks
  )

plot(hist)

qs::qsave(hist, "history.qs")