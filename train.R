
source("flags.R")
import_from("dataset.R", ds, val_ds)
# source("dataset-npz.R")
source("models/base.R")
import_from("models/base.R", model)


callbacks <- list()
# list(
#   callback_reduce_lr_on_plateau(monitor = "loss"),
#   callback_model_checkpoint("model-weights-best-checkpoint.h5", monitor = "loss")
#   # callback_tensorboard(file.path("logs", stringr::str_squish(lubridate::now())))
# )


hist <- model %>% 
  fit(
    ds,
    validation_data = val_ds,
    validation_steps = 10,
    epochs = 100,
    steps_per_epoch = 25,
    callbacks = callbacks
  )

plot(hist)

qs::qsave(hist, "history.qs")