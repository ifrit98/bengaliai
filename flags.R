

FLAGS <- tensorflow::flags(
  flag_integer("height", 137),
  flag_integer("width", 236),
  flag_integer("batch_size", 32),
  flag_integer("val_size", 100),
  flag_string("source_dir", "data/data-tfrecord")
)
