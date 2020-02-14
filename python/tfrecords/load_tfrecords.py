from python.tfrecords.tfrecord_utils import replay_generator
from python.data_tools import crop_resize


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filepath = './data/data-tfrecord' # relative to project_dir

ds_raw = replay_generator(filepath, parallel_files=1)

def test():
  import tensorflow as tf
  it = tf.data.make_one_shot_iterator(ds_raw)
  nb = it.get_next()
  
  sess = tf.Session()
  b  = sess.run(nb)
  print(b)
  
  return b


def view_img(img):
  img = b['image']
  img = 255 - img.reshape(137, 236).astype(np.uint8) # correct inversion
  img = crop_resize(img, size=128)
  fig = plt.subplot()
  plt.imshow(img)
  plt.show()
  
