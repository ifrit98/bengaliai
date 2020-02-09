import os
import itertools
import json
from tempfile import NamedTemporaryFile

import numpy as np
import tensorflow as tf

import string
import random


JSON_FNAME = "TF_RECORD_FEATURES_SPEC.json"


def get_logger():
    import logging
    
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    return logger


def record_generator(batch_generator,
                     output_dir,
                     num_batches, # need a parameter for 'ALL'... Inf?
                     batches_per_file = 25,
                     logger=get_logger(),
                     unbatch = True,
                     test = False):
  """
  Parameters
  batch_generator: instance of a batch generator, must return a flat dict
  output_dir: string, path to output directory
  num_batches: int, number of batches to sink
  batches_per_file: int, number of batches per tfrecord
  
  This will write a json names `FEATURE_DATA.json` to help read the tfrecords
  later using the `replay_generator` function.
  Will also write `ceil(num_batches / batches_per_file)` tfrecords
  
  Usage:
  >>> bg = batch_generator(batch_size=32)
  >>> print(type(bg.next()))
        `dict`
  >>> record_generator(bg, 'path/to/tfrecords/', batches_per_file=50)
  >>> files = os.listdir('path/to/tfrecords/')
  >>> for f in files:
  >>>   print(f)
      'FEATURE_DICT.json'
      tmpaqi23x.tfrecord
      tmpu9cb12.tfrecord
      ...
  """
  output_dir = os.path.abspath(output_dir)
  
  # check if json feature dict already exists in dir
  json_path = os.path.join(output_dir, JSON_FNAME)
  
  # load existing json if exists
  if os.path.isfile(json_path):
    with open(json_path) as f:
      json_dict = json.load(f)
  else:
    batch = next(batch_generator)
    batch = batch_to_nparray(batch)
    batch_generator = itertools.chain([batch], batch_generator)
    if unbatch:
      batch = {k: v[0] for k, v in batch.items()}
      batch['image'] = np.expand_dims(batch['image'], 0) # FIX THIS ELEGANTLY! Return shape (1, x)?
    json_dict = _build_json_dict(batch)
    with open(os.path.join(output_dir, JSON_FNAME), "w") as json_file:
      json.dump(json_dict, json_file)
    
  # e.g. if batches_per_file = 25, num_batches = 103,
  # we get [25, 25, 25, 25, 3]
  batches_per_file = itertools.chain(
    itertools.repeat(batches_per_file, int(num_batches/batches_per_file)),
    [num_batches % batches_per_file])
    
  def validate_shape(batch):
    if unbatch:
      batch_sizes = [len(x) for x in batch.values()]
      assert len(set(batch_sizes)) == 1, "Not all batch dims match"
      record = {k: v[0] for k, v in batch.items()}
      _assert_item_has_expected_shape(record, json_dict, unbatch)
    else:
      _assert_item_has_expected_shape(batch, json_dict, unbatch)
      
  def unbatch_elements(batch):
    """ Returns a numpy array of unbatched records as dicts"""
    return np.asarray([{k: v[i] for k,v in batch.items()}
                                for i in range(batch_size)])
  
  examples = list()
  file_no = 0
  for n in batches_per_file:
    file_no += 1
    output_record = NamedTemporaryFile(dir=output_dir,
                                       suffix='.tfrecord',
                                       delete=False)
    with tf.io.TFRecordWriter(output_record.name) as writer:
      for i, batch in enumerate(batch_generator):
        logger.info(
          'Processing batch: %s of %s in file %s', i, n, file_no)
        if i == n: break # Reached maximum batches_per_file
          
        # coerce records to numpy arrays
        batch = batch_to_nparray(batch)
        validate_shape(batch)
        
        batch_size = next(iter(batch.values())).shape[0]
        
        if test:
          examples.append(batch)
        
        if unbatch:
          batch = unbatch_elements(batch)
          for j in range(batch_size):
            _assert_item_has_expected_shape(batch[j],
                                            json_dict,
                                            unbatch)
            _write_to_tfrecord(batch[j], writer, batch_size=1)
            
        else:
          _write_to_tfrecord(batch, writer, batch_size)
          
  return np.asarray(examples) if test else None
    

# TODO: work on getting replay generator up and running
def replay_generator(src_dir, parallel_files=1):
  """
  Parameters
  src_dir: Directory containing tfrecords to be read into memory
  parallel_files: Number of files to read concurrently
  
  Reads in tensorflow records and parses examples
  Usage:
  >>> dataset = replay_generator('/my/tfrecords/path', 1)
  >>> sess = tf.Session()
  >>> it = dataset.make_one_shot_iterator()
  >>> nb = it.get_next()
  >>> type(sess.run(nb))
      <type 'dict'>
  """
  
  # find json file in directory
  src_dir =os.path.abspath(src_dir)
  json_path = os.path.join(src_dir, JSON_FNAME)
  assert os.path.isfile(json_path)
  
  # bild feature dictionary from json
  READ_FNCS = {
    'FixedLen': tf.io.FixedLenFeature,
    'FixedLenSequence': tf.io.FixedLenSequenceFeature
  }
  
  STR_TO_DTYPE = {
    'tf.float32': tf.float32,
    'tf.int32': tf.int32,
    'tf.int64': tf.int64,
    'tf.string': tf.string
  }

  with open(json_path, "r") as json_file:
    feature_dicts = json.load(json_file)
    
  features = {}
  
  for field_key in feature_dicts.keys():
    # remove odd 'u' from keys as a result of json
    field_key = str(field_key)
    for sub_key in feature_dicts[field_key].keys():
      sub_key = str(sub_key)
    
    fnc_str = feature_dicts[field_key]['fnc']
    dtype_str = feature_dicts[field_key]['dtype']
    shape = feature_dicts[field_key]['shape']
    
    fnc = READ_FNCS[fnc_str]
    dtype = STR_TO_DTYPE[dtype_str]
    
    if fnc.__name__ == 'FixedLenSequenceFeature':
      features[field_key] = fnc(shape, dtype, allow_missing=True)
    else:
      features[field_key] = fnc(shape, dtype)
      
  dataset = tf.data.Dataset.list_files(src_dir + "/*.tfrecord")
  dataset = dataset.interleave(tf.data.TFRecordDataset,
  cycle_length=parallel_files)
  
  # Parse the input `tf.Example` proto using the features dictionary.
  def _parse_example(example_proto):
    return tf.io.parse_single_example(example_proto, features)
  
  return dataset.map(_parse_example)
  

def _assert_batch_has_expected_shape(batch, expected_shape, unbatch):
  assert isinstance(batch, dict)
  
  b_keys = batch.keys()
  e_keys = expected_shape.keys()
  if set(b_keys) != set(e_keys):
    raise ValueError("Batch keys do not match json dictionary")
  for key in b_keys:
    if type(batch[key]) is not np.ndarray and type(batch[key]) is not np.array:
      raise TypeError(
        "Batch is a singular item, not np array ... maybe you meant to set unbatch=False?")
    if batch[key].dtype.name != expected_shape[key]['previous_dtype']:
      raise ValueError(
        "Batch feature datatypes do not match json dictionary")
        
    # exclude batch dim
    if list(batch[key].shape[1:]) != list(expected_shape[key]['shape']):
      raise ValueError(
        "Batch feature shapes do not match json dictionary")

  return True


def _assert_item_has_expected_shape(item, expected_shape, unbatch):
  assert isinstance(item, dict)
  
  r_keys = item.keys()
  e_keys = expected_shape.keys()
  if set(r_keys) != set(e_keys):
    raise ValueError("Item key does not batch json dictionary")
  for key in r_keys:
    if item[key].dtype.name != expected_shape[key]['previous_dtype']:
      raise ValueError(
        "Item feature datatype does not batch json dictionary")
    item_shape = item[key].shape if unbatch else item[key].shape[1:]
    if list(item_shape) != list(expected_shape[key]['shape']):
      print("key:", key)
      print("item_shape:", list(item_shape))
      print("expected_shape:", expected_shape[key]['shape'])
      raise ValueError(
        "Item shape does not match json dictionary")
        
  return True


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))\

def _complex_feature(value):
    """Returns a float list of length len(value)*2 from a complex array"""
    value = value.astype(np.complex64)
    temp = value.view(np.float32)

    return tf.train.Feature(float_list=tf.train.FloatList(value=temp))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


FEAT_FNC_SWITCH = {
    "string":  _bytes_feature,
    "byte":    _bytes_feature,
    "object":  _bytes_feature,

    "float32": _float_feature,
    "float64": _float_feature,
    "complex64":  _complex_feature,
    "complex128": _complex_feature,

    "bool":   _int64_feature,
    "enum":   _int64_feature,
    "int16":  _int64_feature,
    "int32":  _int64_feature,
    "uint32": _int64_feature,
    "int64":  _int64_feature,
    "uint64": _int64_feature
}


def _build_features(feature_dict, keys):
    features = {}
    for i in range(len(feature_dict)):
        dtype = feature_dict[keys[i]].dtype.name
        features[keys[i]] = FEAT_FNC_SWITCH[dtype](
            feature_dict[keys[i]].ravel())
    return features


def _write_to_tfrecord(batch, writer, batch_size=1):
    # TODO: check if this works with unbatched?  e.g. Possibly will send 32331 instead of 8
    NUM_SAMPS = batch_size # next(iter(batch.value()).shape[0]
    keys = list(batch.keys())
    
    for i in range(NUM_SAMPS):
        observation = {k: (v if batch_size == 1 else v[i]) for k, v in batch.items()} 
        features = _build_features(observation, keys)
        example = tf.train.Example(
            features=tf.train.Features(feature=features))
        s = example.SerializeToString()
        writer.write(s)


def _build_json_dict(batch):
    assert isinstance(batch, dict)
    
    batch_keys = batch.keys()
    tensor_metadata = ['fnc', 'dtype', 'shape', 'previous_dtype']
    features = dict.fromkeys(batch_keys)

    for field in batch_keys:
        features[field] = dict.fromkeys(tensor_metadata)

        # excluding batch dim
        if len(batch[field].shape[1:]) == 1 and batch[field].shape[1] == 1:
            features[field]['fnc'] = 'FixedLen'
        else:
            features[field]['fnc'] = 'FixedLenSequence'

        # get shape
        features[field]['shape'] = batch[field].shape[1:]  # exclude batch dim

        # get previous dtype
        features[field]['previous_dtype'] = batch[field].dtype.name
        
        if 'complex' in features[field]['previous_dtype']:
            # complex128 will get casted to complex64 later
            features[field]['previous_dtype'] = 'complex64'

        # get fnc to use while reading
        fnc = FEAT_FNC_SWITCH[batch[field].dtype.name]
        
        # get dtype
        if 'float' in fnc.__name__:
            dtype = 'tf.float32'
        elif 'complex' in fnc.__name__:
            dtype = 'tf.float32'
        elif 'int' in fnc.__name__:
            dtype = 'tf.int64'
        elif 'bytes' in fnc.__name__:
            dtype = 'tf.string'
        else:
            raise ValueError('Cant find datatype')
        
        features[field]['dtype'] = dtype

    return features


def batch_to_nparray(batch):
    np_batch = { k: np.asarray(v) for k,v in batch.items() }
    return np_batch






# Process:
# parquet -> pandas -> numpy -> batch generator -> tfrecord -> tfdataset

import pandas as pd
import numpy as np

import itertools

# Assumes cwd is project_dir (e.g. ~/internal/bengaliai)
# from python.tfrecords.tfrecord_utils import record_generator
# from python.data_tools import normalize_simple


train_df_    = pd.read_csv('data/data-raw/train.csv')
test_df_     = pd.read_csv('data/data-raw/test.csv')
class_map_df = pd.read_csv('data/data-raw/class_map.csv')

NO_VOWELS     = len(train_df_['vowel_diacritic'].unique())
NO_CONSONANTS = len(train_df_['consonant_diacritic'].unique())
NO_GRAPHEMES  = len(train_df_['grapheme_root'].unique())

IMG_COLS = [str(i) for i in range(32332)]

# TODO: add transformations to generator so training dataset is deterministic
# Potential speed gain but will eat more Gb.
def apply_transformations():
  """See augmix.py and data_tools.py"""
  pass


def normalize_simple(image, maximum=255.0, v2=True):
  if v2 is True and len(image.shape) == 1:
    image = [image]
  
  normalized = np.asarray(map(lambda x: (x - np.mean(x)) / np.std(x), image)) \
    if v2 else (maximum - image).astype(np.float64) / maximum
  
  return normalized


# TOOD: add crop/resize/augmentation here
def img_generator(df, norm=False, scale=True, batch_size=8):
  
  for i in range(0, len(df), batch_size):
    image     = np.stack(df.iloc[i:i+batch_size][IMG_COLS].astype(np.int16).values, 0)
    grapheme  = np.stack(df.iloc[i:i+batch_size]['grapheme_root'].astype(np.int16), 0)
    vowel     = np.stack(df.iloc[i:i+batch_size]['vowel_diacritic'].astype(np.int16), 0)
    consonant = np.stack(df.iloc[i:i+batch_size]['consonant_diacritic'].astype(np.int16), 0)
    
    batch = {'image':     normalize_simple(image, scale=scale) if norm else image,
             'grapheme':  np.eye(NO_GRAPHEMES, dtype=np.int16)[grapheme],
             'vowel':     np.eye(NO_VOWELS, dtype=np.int16)[vowel],
             'consonant': np.eye(NO_CONSONANTS, dtype=np.int16)[consonant]}
    yield batch


def create_generators(chain=True, rng=4):
  generators = []
  
  for i in range(rng):
    train_df = pd.merge(
    pd.read_parquet(f'data/data-raw/train_image_data_{i}.parquet'),
    train_df_, on='image_id'
    ).drop(['image_id'], axis=1)
    
    gen = img_generator(train_df)
    generators.append(gen)
    
  return generators if not chain else itertools.chain(generators)

  
## PROCESS:
# Call tfrecord apparatus on list of generators iteratively
# Apply image augmentation: before tfrecord generation or after, (lazily upon tfdataset creation)
# Split up into train/test data.  The "test_df_" is left blank intentionally. No labels.

def make_tfrecords(outdir='/tmp/tfrecords', chain=True, rng=1, num_batches=6276):
  ## Create tfdataset
  generators = create_generators(chain, rng=rng)
  
  if not chain:
    for gen in generators:
      record_generator(batch_generator=gen, output_dir=outdir, num_batches=num_batches)
  else:
    record_generator(batch_generator=generator, output_dir=outdir, num_batches=num_batches)
  
  print("Finished creating tfrecords in %s", outdir)
    
  return True


output_dir = src_dir = '/home/jason/internal/bengali/data/data-tfrecord'
batch_generator = create_generators(False, 1)[0]
num_batches = 100
unbatch = True
batch = next(batch_generator)
batch = batch_to_nparray(batch)
batch_generator = itertools.chain([batch], batch_generator)
if unbatch:
  batch = {k: v[0] for k, v in batch.items()}
  batch['image'] = np.expand_dims(batch['image'], 0) # FIX THIS ELEGANTLY! Return shape (1, x)?
json_dict = _build_json_dict(batch)

print(json_dict) # (32332,) upon build_json

with open(os.path.join(output_dir, JSON_FNAME), "w") as json_file:
  json.dump(json_dict, json_file)


