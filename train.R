

if (!exists_here("FLAGS")) {
  import_from("flags.R", FLAGS)
}

# source("dataset-npz.R")
import_from("dataset.R", ds, val_ds)
# TODO: Add find model functionality from JSGutils/deepR
# import_from("models/base.R", model)
# import_from("models/conv-simple.R", model)
# import_from("models/conv-small.R", model)
import_from("models/SE_resnet50.R", model)


callbacks <-
  list(
    callback_reduce_lr_on_plateau(monitor = "grapheme_root_loss"),
    callback_reduce_lr_on_plateau(monitor = "consonant_loss"),
    callback_reduce_lr_on_plateau(monitor = "vowel_loss"),
    callback_early_stopping(monitor = "grapheme_root_loss", patience = 25),
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

hist <- qs::qread("history-SEnet.qs")

# TODO: investigate images after reading TFRECORDS -> MAKE SURE THEY ARE LEGIT
# TODO: Try CLR in different modes.  [Triangular, Exponential]
# TODO: Add tfruns functionality to launch from terminal**
# TODO: write out flags/metadata from JSGutils/deepR for each run
# TODO: Make much bigger dataset by augmentation/translation before serializing to tfrecord
# TODO: try dynamic capsule routing network model
# TODO: implement CLR for this project
# TODO: implement learning rate and momentum range tests, as well as learning curves from Andrew Ng
## error vs LR
## error vs # params (model complexity)
## error vs lambda (weight decay, dropout)
## error vs training set size (turn on-off augmentations)
## F1 score for unbalanced classes
# TODO: weighting schemes for roots, vowels, consonants??