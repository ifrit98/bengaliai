from neid_dl.train.dataset_utils import read_dataset
import tensorflow as tf
# os.uname global var



def batch_generator(fp):
  sess = tf.Session()
  ds = get_dataset()
  it = ds.make_one_shot_iterator()
  nb = it.get_next()
  return nb
  
  def get_next():
    while True:
      return sess.run(nb)
  
  return get_next()

gen = batch_generator(fp) 
b = get_next()


def import_data(filepath=None):
  filepath = filepath or LOCAL_FP
  return read_dataset(filepath)

def get_dataset(filepath=None,
                batch_size=8,
                prefetch=False):

  ds = import_data(filepath)
  {k: v for k, v in dict}
  [x for x in z]
  (x for x in z) # returns an iterator
  # Dictionary comprehension
  ds = ds.map(lambda x: {k: x[k] for k in  ['predicted_prob_symbol',
    'message_length',
    'predicted_prob_is_transition',
    'message',
    'sample_symbol_class',
    'sample_is_symbol_transition'] if k in x.keys()})

# Intersection
my_keys & x.keys()

# No separate var for batch (if batch is not none)
  if batch:
    ds = ds.batch(batch_size, drop_remainder=True)

  return ds.prefetch(8) if prefetch else ds

# Specific classname
class SingleBatchGenerator:
  """ Simple generator class to hide details from end user.
      Parses and prepares dataset with or without batching, has exposed callable
      next() to provide a single dataset item after sess.run(nb).
    Parameters
      __init__
        filepath: Path to tfrecords (must include JSON description file in dir)
          (Default: None)
        winston: (bool) Whether or not to use Winston copy of demod tfrecords
          (default: False)
        full: (bool) Use full dataset or a subset (i.e. points to hardcoded dir)
          (default: True)
        batch: (bool) Whether to batch dataset (default: False)
        batch_size: (int) Number of records per batch (default: 8)
      Usage:
        ```from generator import Generator

           gen = Generator()
           batch = gen.next()
           print(batch)
        ```

             {'message_length': array([1078.], dtype=float32),
             'predicted_prob_is_transition': array([[2.9802322e-07, ...,
                     1.9440782e-01, 1.7805150e-01]], dtype=float32),
             'sample_is_symbol_transition': array([[0., ..., 0.]], dtype=float32)}
             ...
  """

  def __init__(self, 
               filepath=None, 
               full=True, 
               batch_size=8):
    if ONWINSTON and not filepath:
      if full:
        self.filepath = WINSTON_FP
      else:
        self.filepath = WINSTON_SM
    elif filepath is None:
      self.filepath = LOCAL_FP
    else:
      self.filepath = filepath

    self.batch = batch
    self.batch_size = batch_size
    self.generator = self.get_generator_iter()
    self.sess = None
    self.nb = None
  
  def ensure_initialized():
    # if set, good to go
    # if not, init everything required
    pass
  
  def next(self):
    return self.sess.run(self.nb)

  def get_session(self):
    if self.sess is None:
        self.sess = tf.Session()
    return self.sess
    
  def init_iterator(self):
    sess = self.get_session()
    ds = create_dataset(batch=self.batch, batch_size=self.batch_size)
    it = ds.make_one_shot_iterator()
    self.nb = it.get_next()
    

  def get_generator_iter(self):

    while True:
      try:
        yield sess.run(nb)

      except tf.errors.OutOfRangeError:
        print("Dataset exhausted.")
        break

  def get_batch(self):
    return self.generator.next()

