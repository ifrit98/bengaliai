#!/usr/bin/python


import tensorflow as tf
import pandas as pd
from data_tools import crop_resize

HEIGHT = 137
WIDTH  = 236
BATCH  = 32
RESIZE = 128

filename = 'train{0}.tfrecord'

# Load data
train_md = pd.read_csv('data-raw/train.csv')
test_md = pd.read_csv('data-raw/test.csv')
class_map_df = pd.read_csv('data-raw/class_map.csv')
sample_sub_df = pd.read_csv('data-raw/sample_submission.csv')



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



def serialize_example(image, labels):
  image_string = tf.io.serialize_tensor(image)
  feature = {
    'height':          _int64_feature(HEIGHT), # 137
    'width':           _int64_feature(WIDTH),  # 236
    'depth':           _int64_feature(1),      # 1
    'label_grapheme':  _int64_feature(labels[0]),
    'label_vowel':     _int64_feature(labels[1]),
    'label_consonant': _int64_feature(labels[2]),
    'image_raw':       _bytes_feature(image_string),
  }
  proto = tf.train.Example(features=tf.train.Features(feature=feature))
  return proto.SerializeToString()
  
  

def tf_serialize_image(image_string, labels):
  tf_string = tf.py_function(
    serialize_image,
    (image_string, labels),
    tf.string)
  return tf.reshape(tf_string, ())
  

# Training data
for i in range(len(TRAIN)):
  train = pd.read_parquet(TRAIN[i]).drop(columns=['image_id'])
  with tf.io.TFRecordWriter(filename.format(i)) as writer:
    for idx, row in train.iterrows():
      img = row.values.reshape([HEIGHT, WIDTH, 1])
      labs = train_md.iloc[int(idx + (i * len(train)))].values[1:-1] # root, vowel, consonant
      serialized_ex = serialize_example(img, labs)
      writer.write(serialized_ex)
    
# Test data
for i in range(len(TEST)):
  test = pd.read_parquet(TEST[i]).drop(columns=['image_id'])
  with tf.io.TFRecordWriter(filename.format(i)) as writer:
    for idx, row in test.iterrows():
      crop_resize(row.values, pad=0)
      img = row.values.reshape([HEIGHT, WIDTH, 1])
      labs = train_md.iloc[int(idx + (i * len(train)))].values[1:-1] # root, vowel, consonant
      serialized_ex = serialize_example(img, labs)
      writer.write(serialized_ex)
    
    

i = 1
test = pd.read_parquet(TEST[i]).drop(columns=['image_id'])
x = test.iloc[i].values
x = x.reshape([HEIGHT, WIDTH])
img = crop_resize(x, size=RESIZE, size2= 75, pad=0)
df = resize(test, RESIZE)
im2 = df.iloc[i].values
im2 = im2.reshape([RESIZE, RESIZE])

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(x)
plt.show()


# Scale AND resize
plt.figure()
plt.imshow(img)
plt.show()

# Only resize
plt.figure()
plt.imshow(im2)
plt.show()
