import tensorflow as tf

HEIGHT = 137
WIDTH  = 236

filename = "data/data-tfrecord/train0.tfrecord"

# Read back in to validate
filenames = [filename]
raw_dataset = tf.data.TFRecordDataset(filenames)


# Create a description of the features.
feature_description = {
    'height': tf.io.FixedLenFeature([], tf.int64, default_value=HEIGHT),
    'width': tf.io.FixedLenFeature([], tf.int64, default_value=WIDTH),
    'depth': tf.io.FixedLenFeature([], tf.int64, default_value=1),
    'label_grapheme': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'label_vowel': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'label_consonant': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image_raw': tf.io.FixedLenFeature([], tf.string, default_value=''),
}


# Parse the input `tf.Example` proto using the dictionary above.
def _parse_example(example_proto):
  return tf.io.parse_single_example(example_proto, feature_description)


def parse_image(x):
  x['image_raw'] = tf.io.parse_tensor(x['image_raw'], tf.uint8)
  return x


# First parse example
ds_raw = raw_dataset.map(_parse_example)

# Parse image tensor (can we do this in one step?)
ds_parsed = ds_raw.map(parse_image)


# it2 = tf.compat.v1.data.make_one_shot_iterator(ds2)

# for i in range(10):
#   b = it2.get_next()
#   labels = np.asarray([b['label_consonant'], b['label_vowel'], b['label_grapheme']])
#   print(labels)
