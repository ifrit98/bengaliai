IMGSIZE = 128

FLAGS <- tensorflow::flags(
  flag_integer("epochs", 200),
  flag_integer("steps_per_epoch", 250),
  flag_integer("height", IMGSIZE),
  flag_integer("width", IMGSIZE),
  flag_integer("batch_size", 32),
  flag_integer("val_size", 100),
  flag_string("source_dir", "data/data-tfrecord-norm/")
)
