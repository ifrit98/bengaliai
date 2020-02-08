import tensorflow as tf

EAGER = "<class 'tensorflow.python.framework.ops.EagerTensor'>"


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))) and str(type(value)) == EAGER:
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))



# TODO: Format incoming image either via tf.io.serialize_tensor or ...?
def serialize_image(image, labels):
  
  image_string = tf.io.serialize_tensor(image)
  # lables = [tf.io.serialize_tensor(l) for l in labels]
  
  feature = {
    'height':          _int64_feature(137), # 137
    'width':           _int64_feature(236),  # 236
    'depth':           _int64_feature(1),  # 1
    'label_consonant': _int64_feature(labels[0]),
    'label_vowel':     _int64_feature(labels[1]),
    'label_grapheme':  _int64_feature(labels[2]),
    # 'label_consonant': _bytes_feature(labels[0]),
    # 'label_vowel':     _bytes_feature(labels[1]),
    # 'label_grapheme':  _bytes_feature(labels[2]),
    'image_raw':       _bytes_feature(image_string),
  }
  proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return proto.SerializeToString() # tf.train.Example(features=tf.train.Features(feature=feature))
  


def tf_serialize_image(image_string, labels):
  tf_string = tf.py_function(
    serialize_image,
    (image_string, labels),
    tf.string)
  return tf.reshape(tf_string, ())
  
  
#   
# def write_tfrecords(record_file='images.tfrecords'):
#   with tf.io.TFRecordWriter(record_file) as writer:
#     for filename, label in image_labels.items():
#       image_string = open(filename, 'rb').read()
#       tf_example = serialize_image(image_string, label)
#       writer.write(tf_example.SerializeToString())
# 
# V2 writer
# def write_tfrecordsv2():
#   tfrecord_dir = 'tfrecords/data.tfrecords'
#   with tf.io.TFRecordWriter(tfrecord_dir) as writer:
#       for image_path, label in zip(image_paths, labels):
#           
#           img = tf.keras.preprocessing.image.load_img(image_path)
#           img_array = tf.keras.preprocessing.image.img_to_array(img)
#           
#           img_array = tf.keras.preprocessing.image.random_zoom(img_array, (0.5,0.5),
#                                            row_axis=0,
#                                            col_axis=1,
#                                            channel_axis=2)
#           
#           img_bytes = tf.io.serialize_tensor(img_array)
#           image_shape = img_array.shape
#           
#           example = serialize_example(img_bytes, label, image_shape)
#           writer.write(example)
