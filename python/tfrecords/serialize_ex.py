# import tensorflow as tf
# 
# 
# # The following functions can be used to convert a value to a type compatible
# # with tf.Example.
# 
# def _bytes_feature(value):
#   """Returns a bytes_list from a string / byte."""
#   if isinstance(value, type(tf.constant(0))):
#     value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
#   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
# 
# def _float_feature(value):
#   """Returns a float_list from a float / double."""
#   return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
# 
# def _int64_feature(value):
#   """Returns an int64_list from a bool / enum / int / uint."""
#   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
# 
# 
# 
# cat_in_snow  = tf.keras.utils.get_file('320px-Felis_catus-cat_on_snow.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg')
# williamsburg_bridge = tf.keras.utils.get_file('194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/194px-New_East_River_Bridge_from_Brooklyn_det.4a09796u.jpg')
# 
# image_labels = {
#     cat_in_snow : 0,
#     williamsburg_bridge : 1,
# }
# 
# # This is an example, just using the cat image.
# image_string = open(cat_in_snow, 'rb').read()
# 
# label = image_labels[cat_in_snow]
# sess = tf.compat.v1.Session()
# image = sess.run(tf.image.decode_jpeg(image_string))
# image_shape = image.shape
# 
# def image_example(image_string, label):
#   image_shape = tf.image.decode_jpeg(image_string).shape
#   
#   feature = {
#     'height': _intfeatu64_feature(image_shape[0]), # 137
#     'width': _int64_feature(image_shape[1]),  # 236
#     'depth': _int64_feature(image_shape[2]),  # 1
#     'label': _int64_feature(label),
#     'image_raw': _bytes_feature(image_string),
#   }
#   return tf.train.Example(features=tf.train.Features(feature=feature))
#   
# 
# example = image_example(image_string, label)
#   
# for line in str(image_example(image_string, label)).split('\n')[:5]:
#   print(line)
# print("...")
# 
# 
# # Write the raw image files to `images.tfrecords`.
# # First, process the two images into `tf.Example` messages.
# # Then, write to a `.tfrecords` file.
# record_file = 'images.tfrecords'
# with tf.io.TFRecordWriter(record_file) as writer:
#   for filename, label in image_labels.items():
#     image_string = open(filename, 'rb').read()
#     tf_example = image_example(image_string, label)
#     writer.write(tf_example.SerializeToString())
# 
# 
# 
# # Read back in
# raw_image_dataset = tf.data.TFRecordDataset('images.tfrecords')
# 
# # Create a dictionary describing the features.
# image_feature_description = {
#     'height': tf.io.FixedLenFeature([], tf.int64),
#     'width': tf.io.FixedLenFeature([], tf.int64),
#     'depth': tf.io.FixedLenFeature([], tf.int64),
#     'label': tf.io.FixedLenFeature([], tf.int64),
#     'image_raw': tf.io.FixedLenFeature([], tf.string),
# }
# 
# def _parse_image_function(example_proto):
#   # Parse the input tf.Example proto using the dictionary above.
#   return tf.io.parse_single_example(example_proto, image_feature_description)
# 
# parsed_image_dataset = raw_image_dataset.map(_parse_image_function)
# parsed_image_dataset
# 
# 
# for image_features in parsed_image_dataset:
#   image_raw = image_features['image_raw'].numpy()
#   display.display(display.Image(data=image_raw))
