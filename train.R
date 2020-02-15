

if (!exists_here("FLAGS")) {
  import_from("flags.R", FLAGS)
}

# source("dataset-npz.R")
import_from("dataset.R", ds, val_ds)
import_from("models/base.R", model)


callbacks <- #list()
  list(
    callback_reduce_lr_on_plateau(monitor = "grapheme_root_loss"),
    callback_reduce_lr_on_plateau(monitor = "consonant_loss"),
    callback_reduce_lr_on_plateau(monitor = "vowel_loss"),
    callback_model_checkpoint("model-weights-best-checkpoint.h5", monitor = "grapheme_root_acc"),
    callback_tensorboard(file.path("logs", stringr::str_squish(lubridate::now())))
  )


hist <- model %>% 
  fit(
    ds,
    validation_data = val_ds,
    validation_steps = FLAGS$val_size,
    epochs = FLAGS$epochs,
    steps_per_epoch = FLAGS$steps_per_epoch,
    callbacks = callbacks
  )

plot(hist)

qs::qsave(hist, "history.qs")

# TODO: investigate images after reading TFRECORDS -> MAKE SURE THEY ARE LEGIT