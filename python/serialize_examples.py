
import tensorflow as tf
import numpy as np


# The following functions can be used to convert a value to a type compatible
# with tf.Example.

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# TODO: Format incoming image either via tf.io.serialize_tensor or ...?
def image_example(image_string, labels):
  
  feature = {
    'height':          _int64_feature(137), # 137
    'width':           _int64_feature(236),  # 236
    'depth':           _int64_feature(1),  # 1
    'label_consonant': _int64_feature(labels[0]),
    'label_vowel':     _int64_feature(labels[1]),
    'label_grapheme':  _int64_feature(labels[2]),
    'image_raw':       _bytes_feature(image_string),
  }
  return tf.train.Example(features=tf.train.Features(feature=feature))
  



def serialize_example(feature0, feature1, feature2, feature3):
  """
  Creates a tf.Example message ready to be written to a file.
  """
  # Create a dictionary mapping the feature name to the tf.Example-compatible
  # data type.
  feature = {
      'feature0': _int64_feature(feature0),
      'feature1': _int64_feature(feature1),
      'feature2': _bytes_feature(feature2),
      'feature3': _float_feature(feature3),
  }
  # Create a Features message using tf.train.Example.
  example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return example_proto.SerializeToString()



print(_bytes_feature(b'test_string'))
print(_bytes_feature(u'test_bytes'.encode('utf-8')))

print(_float_feature(np.exp(1)))

print(_int64_feature(True))
print(_int64_feature(1))


feature = _float_feature(np.exp(1))
feature.SerializeToString()


## Example tf.Example
# The number of observations in the dataset.
n_observations = int(1e4)

# Boolean feature, encoded as False or True.
feature0 = np.random.choice([False, True], n_observations)

# Integer feature, random from 0 to 4.
feature1 = np.random.randint(0, 5, n_observations)

# String feature
strings = np.array([b'cat', b'dog', b'chicken', b'horse', b'goat'])
feature2 = strings[feature1]

# Float feature, from a standard normal distribution
feature3 = np.random.randn(n_observations)



example_observation = []

serialized_example = serialize_example(False, 4, b'goat', 0.9876)
serialized_example

# Decoding
example_proto = tf.train.Example.FromString(serialized_example)
example_proto

"""Note: 
  There is no requirement to use tf.Example in TFRecord files. 
  tf.Example is just a method of serializing dictionaries to byte-strings. 
  Lines of text, encoded image data, or serialized tensors (using tf.io.serialize_tensor, 
  and tf.io.parse_tensor when loading). See the tf.io module for more options.
"""
tf.io.serialize_tensor()




def tf_serialize_example(f0,f1,f2,f3):
  tf_string = tf.py_function(
    serialize_example,
    (f0,f1,f2,f3),  # pass these args to the above function.
    tf.string)      # the return type is `tf.string`.
  return tf.reshape(tf_string, ()) # The result is a scalar
  
  
features_dataset = tf.data.Dataset.from_tensor_slices((feature0, feature1, feature2, feature3))
features_dataset
  
  
tf_serialize_example(f0,f1,f2,f3)

# Apply this function to each element in the dataset:
serialized_features_dataset = features_dataset.map(tf_serialize_example)
serialized_features_dataset  


def generator():
  for features in features_dataset:
    yield serialize_example(*features)


serialized_features_dataset = tf.data.Dataset.from_generator(
    generator, output_types=tf.string, output_shapes=())

serialized_features_dataset

# Write to tfrecord
filename = 'test.tfrecord'
writer = tf.data.experimental.TFRecordWriter(filename)
writer.write(serialized_features_dataset)
