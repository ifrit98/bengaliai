

FLAGS <- tensorflow::flags(
  flag_integer("height", 64),
  flag_integer("width", 64),
  flag_integer("batch_size", 2),
  flag_integer("val_size", 10),
  flag_string("source_dir", "data/data-tfrecord")
)
