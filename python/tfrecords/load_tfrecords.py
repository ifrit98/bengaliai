from python.tfrecords.tfrecord_utils import replay_generator

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
