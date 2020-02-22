IMGSIZE = 128

FLAGS <- tensorflow::flags(
  flag_integer("epochs", 250),
  flag_integer("steps_per_epoch", 250),
  flag_integer("height", IMGSIZE),
  flag_integer("width", IMGSIZE),
  flag_integer("batch_size", 32),
  flag_integer("val_size", 100),
  flag_boolean("global_pool", TRUE),
  flag_string("source_dir", "data/data-tfrecord-norm"),
  flag_string("model", "SE_resnet50"),
  flag_integer("no_blocks", 3),
  flag_integer("patience", 40),
  flag_integer("valid_freq", 3)
)

FLAGS$model <- sprintf("models/%s.R", FLAGS$model)
