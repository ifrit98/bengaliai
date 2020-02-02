from dataset_utils import read_dataset
import tensorflow as tf

# WINSTON_SM = "/mnt/md0/georgej/sample-classifier-predictions/4fsk"
# WINSTON_FP = "/mnt/md0/demod/sample-classifier-predictions/4fsk/2019-05-23T21-03-26.661399Z"
# LOCAL_FP = "/mnt/ipahome/georgej/dev/sample-classifier-predictions/4fsk"


def import_data(filepath=None):
  filepath = filepath or LOCAL_FP
  return read_dataset(filepath)

def get_dataset(filepath=None,
                batch_size=8,
                prefetch=False):

  ds = import_data(filepath)
        
  return ds


def batch_generator(fp):
  sess = tf.Session()
  ds = get_dataset()
  it = ds.make_one_shot_iterator()
  nb = it.get_next()
  
  def get_next():
    while True:
      return sess.run(nb)

  return get_next
  
gen = batch_generator(LOCAL_FP) 

for _ in range(10):
  b = gen()
  print(b)
